# Networks/model.py
import os, math, glob, numpy as np
from collections import defaultdict
from PIL import Image

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.cluster import MiniBatchKMeans

from Dataset.dataLoader import make_loaders, CLASSES, TILE_H, TILE_W
from Dataset.makeGraph import (
    plot_curves, plot_confusion, plot_bars, plot_double_bars
)
from utils import (
    colorize, stitch_tiles, stitch_tiles_rgb, overlay_segmentation,
    save_png, miou_score, upsample_nearest, save_json, fast_hist,
    parse_rr_cc
)

# Architectures
from Networks.Architectures.unet_plus import UNetPlus
from Networks.Architectures.unet import UNet  # utilisé aussi pour le SSL (autoencodeur)
from Networks.Architectures.basicNetwork import BasicNet  # fallback éventuel

CLASS_NAMES = ["others", "water", "buildings", "farmlands", "green"]


# --------------------------------------------------------------------------------------
#                                     LOSSES
# --------------------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, n_classes: int, eps: float = 1e-6):
        super().__init__()
        self.n = n_classes
        self.eps = eps

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target, num_classes=self.n).permute(0, 3, 1, 2).float()
        inter = (probs * target_1h).sum(dim=(0, 2, 3))
        union = (probs + target_1h).sum(dim=(0, 2, 3))
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, n_classes: int, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-6):
        super().__init__()
        self.n = n_classes
        self.a = alpha
        self.b = beta
        self.g = gamma
        self.eps = eps

    def forward(self, logits, target):
        p = F.softmax(logits, dim=1)
        t = F.one_hot(target, num_classes=self.n).permute(0, 3, 1, 2).float()
        tp = (p * t).sum(dim=(0, 2, 3))
        fp = (p * (1 - t)).sum(dim=(0, 2, 3))
        fn = ((1 - p) * t).sum(dim=(0, 2, 3))
        tversky = (tp + self.eps) / (tp + self.a * fp + self.b * fn + self.eps)
        return (1 - tversky.mean()) ** self.g


# --------------------------------------------------------------------------------------
#                                  HELPERS / METRICS
# --------------------------------------------------------------------------------------
def scores_from_hist(hist):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    miou = float(np.nanmean(iu))
    pixacc = float(np.diag(hist).sum() / (hist.sum() + 1e-10))
    return miou, pixacc, iu.tolist()


def concat_h(im_left, im_right):
    return np.concatenate([im_left, im_right], axis=1)


def crop_test_quadrant(arr, n_rows, n_cols, tile_h=TILE_H, tile_w=TILE_W):
    r0 = (n_rows // 2) * tile_h
    c0 = (n_cols // 2) * tile_w
    if arr.ndim == 2:
        return arr[r0:, c0:]
    return arr[r0:, c0:, :]


def load_rgb_tiles_by_names(data_root, names):
    out = {}
    img_dir = os.path.join(data_root, "images")
    for nm in names:
        p = os.path.join(img_dir, nm)
        rgb = np.array(Image.open(p).convert("RGB"))
        out[nm] = rgb
    return out


# ======================================================================================
#                                   PART A : SUPERVISED
# ======================================================================================
class SupervisedExperiment:
    def __init__(self, cfg, out_dir, device):
        self.cfg = cfg
        self.out = out_dir
        self.device = device

        # Data
        self.Ltr, self.Lva, self.Lte, self.grid = make_loaders(cfg)

        # Arch
        in_ch = 4 if cfg.get("ADD_EXG", False) else 3
        feats = tuple(cfg.get("UNET_FEATS", [64, 128, 256, 512]))
        arch = cfg.get("ARCH", "unet_plus").lower()
        if arch == "unet_plus":
            self.net = UNetPlus(in_ch=in_ch, out_classes=CLASSES, feats=feats, drop=cfg.get("DROP", 0.1))
        elif arch == "unet":
            self.net = UNet(in_ch=in_ch, out_classes=CLASSES, feats=feats)
        else:
            self.net = BasicNet(in_ch=in_ch, out_classes=CLASSES)
        self.net.to(device)

        # Class weights (auto sur train)
        def _estimate_class_weights(loader):
            counts = np.zeros(CLASSES, dtype=np.float64)
            for _, y, _ in loader:
                y = y.numpy()
                for c in range(CLASSES):
                    counts[c] += (y == c).sum()
            freq = counts / counts.sum()
            # trick anti-explosion
            w = 1.0 / (np.log(1.02 + freq))
            return torch.tensor(w, dtype=torch.float32, device=device)

        if cfg.get("CLASS_WEIGHTS", "auto") == "auto":
            ce_w = _estimate_class_weights(self.Ltr)
        else:
            ce_w = torch.tensor(cfg["CLASS_WEIGHTS"], dtype=torch.float32, device=device)

        # Losses
        self.ce = nn.CrossEntropyLoss(weight=ce_w, label_smoothing=cfg.get("LABEL_SMOOTH", 0.05))
        self.dice = DiceLoss(CLASSES)
        self.ftl = FocalTverskyLoss(
            CLASSES,
            alpha=cfg.get("FTL_ALPHA", 0.3),
            beta=cfg.get("FTL_BETA", 0.7),
            gamma=cfg.get("FTL_GAMMA", 1.33),
        )
        self.lambda_dice = cfg.get("DICE_LAMBDA", 0.5)
        self.lambda_ftl = cfg.get("FTL_LAMBDA", 0.4)

        # Optim / sched / AMP / clip
        self.opt = optim.AdamW(self.net.parameters(), lr=cfg.get("LR", 3e-4), weight_decay=cfg.get("WEIGHT_DECAY", 1e-4))
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.get("EPOCHS", 80))
        self.scaler = GradScaler(enabled=(device.type == "cuda"))
        self.max_norm = cfg.get("GRAD_CLIP", 1.0)
        self.epochs = cfg.get("EPOCHS", 80)

        # Folders
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.cmp_dir, exist_ok=True)

    # --- dirs ---
    @property
    def ckpt_dir(self): return os.path.join(self.out, "checkpoints")
    @property
    def fig_dir(self):  return os.path.join(self.out, "figs")
    @property
    def test_dir(self): return os.path.join(self.out, "test")
    @property
    def cmp_dir(self):  return os.path.join(self.out, "compare")

    # --- train/val step ---
    def _forward_loss(self, x, y, train=True):
        with autocast(enabled=(self.device.type == "cuda")):
            logits = self.net(x)
            loss = self.ce(logits, y) + self.lambda_dice * self.dice(logits, y) + self.lambda_ftl * self.ftl(logits, y)
        if train:
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.max_norm is not None:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
        return loss, logits

    def run(self):
        hist = {"train_loss": [], "val_loss": [], "train_mIoU": [], "val_mIoU": []}
        best, best_path = -1.0, None

        for ep in range(1, self.epochs + 1):
            # ---- TRAIN ----
            self.net.train()
            tsum, tcnt, tmi = 0.0, 0, 0.0
            for x, y, _ in tqdm(self.Ltr, desc=f"Train {ep}/{self.epochs}", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                loss, logits = self._forward_loss(x, y, train=True)
                pred = logits.argmax(1).detach().cpu().numpy()
                miou, _, _ = miou_score(y.cpu().numpy(), pred, CLASSES)
                b = x.size(0); tcnt += b; tsum += loss.item() * b; tmi += miou * b
            self.sched.step()

            # ---- VAL ----
            self.net.eval()
            vsum, vcnt, vmi = 0.0, 0, 0.0
            with torch.no_grad():
                for x, y, _ in tqdm(self.Lva, desc=f"Val {ep}/{self.epochs}", leave=False):
                    x, y = x.to(self.device), y.to(self.device)
                    loss, logits = self._forward_loss(x, y, train=False)
                    pred = logits.argmax(1).cpu().numpy()
                    miou, _, _ = miou_score(y.cpu().numpy(), pred, CLASSES)
                    b = x.size(0); vcnt += b; vsum += loss.item() * b; vmi += miou * b

            hist["train_loss"].append(tsum / max(tcnt, 1))
            hist["val_loss"].append(vsum / max(vcnt, 1))
            hist["train_mIoU"].append(tmi / max(tcnt, 1))
            hist["val_mIoU"].append(vmi / max(vcnt, 1))

            plot_curves(hist, os.path.join(self.fig_dir, "curves.png"))

            if hist["val_mIoU"][-1] > best:
                best = hist["val_mIoU"][-1]
                best_path = os.path.join(self.ckpt_dir, "best.pt")
                torch.save({"ep": ep, "state": self.net.state_dict()}, best_path)

        if best_path:
            self.net.load_state_dict(torch.load(best_path)["state"])

        self.generate_mosaics_and_charts()

    # --- predictions with optional TTA flip ---
    def _predict_loader(self, loader):
        pred_tiles, gt_tiles = {}, {}
        hist_sum = np.zeros((CLASSES, CLASSES), dtype=np.int64)
        tta = self.cfg.get("TTA_FLIP", True)
        with torch.no_grad():
            for x, y, names in tqdm(loader, desc="Predict"):
                x = x.to(self.device)
                logits = self.net(x)
                if tta:
                    logits_f = self.net(torch.flip(x, dims=[-1]))
                    logits = (logits + torch.flip(logits_f, dims=[-1])) / 2
                pred = logits.argmax(1).cpu().numpy()
                y_np = y.cpu().numpy()
                hist_sum += fast_hist(y_np.flatten(), pred.flatten(), CLASSES)
                for i, nm in enumerate(names):
                    pred_tiles[nm] = pred[i].astype(np.uint8)
                    gt_tiles[nm] = y_np[i].astype(np.uint8)
        return pred_tiles, gt_tiles, hist_sum

    # --- reporting: mosaics + overlays + figures ---
    def generate_mosaics_and_charts(self):
        (n_rows, n_cols) = self.grid
        data_root = self.cfg.get("DATA_ROOT", "Dataset")

        # Prédire sur chaque split
        self.net.eval()
        tr_pred, tr_gt, cm_tr = self._predict_loader(self.Ltr)
        va_pred, va_gt, cm_va = self._predict_loader(self.Lva)
        te_pred, te_gt, cm_te = self._predict_loader(self.Lte)

        # Fusion FULL
        pred_all = {}; pred_all.update(tr_pred); pred_all.update(va_pred); pred_all.update(te_pred)
        gt_all = {}; gt_all.update(tr_gt); gt_all.update(va_gt); gt_all.update(te_gt)

        # Label mosaics
        full_pred_lbl = stitch_tiles(pred_all, n_rows, n_cols, TILE_H, TILE_W)
        full_gt_lbl   = stitch_tiles(gt_all,   n_rows, n_cols, TILE_H, TILE_W)
        test_pred_lbl = crop_test_quadrant(full_pred_lbl, n_rows, n_cols)
        test_gt_lbl   = crop_test_quadrant(full_gt_lbl,   n_rows, n_cols)

        full_pred_rgb = colorize(full_pred_lbl)
        full_gt_rgb   = colorize(full_gt_lbl)
        test_pred_rgb = colorize(test_pred_lbl)
        test_gt_rgb   = colorize(test_gt_lbl)

        # RGB mosaics (orthophoto)
        all_names = list(pred_all.keys())
        rgb_tiles = load_rgb_tiles_by_names(data_root, all_names)
        full_rgb  = stitch_tiles_rgb(rgb_tiles, n_rows, n_cols, TILE_H, TILE_W)
        test_rgb  = crop_test_quadrant(full_rgb, n_rows, n_cols)

        # Overlays
        full_overlay_pred = overlay_segmentation(full_rgb, full_pred_lbl)
        full_overlay_gt   = overlay_segmentation(full_rgb, full_gt_lbl)
        test_overlay_pred = overlay_segmentation(test_rgb, test_pred_lbl)
        test_overlay_gt   = overlay_segmentation(test_rgb, test_gt_lbl)

        # Saves (compare/)
        save_png(full_rgb,                 os.path.join(self.cmp_dir, "full_rgb.png"))
        save_png(test_rgb,                 os.path.join(self.cmp_dir, "test_rgb.png"))
        save_png(full_overlay_pred,        os.path.join(self.cmp_dir, "full_overlay_pred.png"))
        save_png(full_overlay_gt,          os.path.join(self.cmp_dir, "full_overlay_gt.png"))
        save_png(test_overlay_pred,        os.path.join(self.cmp_dir, "test_overlay_pred.png"))
        save_png(test_overlay_gt,          os.path.join(self.cmp_dir, "test_overlay_gt.png"))
        save_png(full_pred_rgb,            os.path.join(self.cmp_dir, "full_pred.png"))
        save_png(full_gt_rgb,              os.path.join(self.cmp_dir, "full_gt.png"))
        save_png(test_pred_rgb,            os.path.join(self.cmp_dir, "test_pred.png"))
        save_png(test_gt_rgb,              os.path.join(self.cmp_dir, "test_gt.png"))
        save_png(concat_h(full_rgb, full_overlay_pred), os.path.join(self.cmp_dir, "full_RGB_vs_PRED-overlay.png"))
        save_png(concat_h(test_rgb, test_overlay_pred), os.path.join(self.cmp_dir, "test_RGB_vs_PRED-overlay.png"))

        # Error maps on RGB
        err_full = (full_pred_lbl != full_gt_lbl)
        err_img_full = full_rgb.copy(); err_img_full[err_full] = np.array([255, 0, 0], dtype=np.uint8)
        save_png(err_img_full, os.path.join(self.cmp_dir, "full_errormap_onRGB.png"))
        err_test = (test_pred_lbl != test_gt_lbl)
        err_img_test = test_rgb.copy(); err_img_test[err_test] = np.array([255, 0, 0], dtype=np.uint8)
        save_png(err_img_test, os.path.join(self.cmp_dir, "test_errormap_onRGB.png"))

        # Metrics/plots
        cm_full = cm_tr + cm_va + cm_te
        m_test, acc_test, iu_test = scores_from_hist(cm_te)
        m_full, acc_full, iu_full = scores_from_hist(cm_full)
        save_json({
            "test": {"mIoU": m_test, "pixel_acc": acc_test, "per_class_IoU": iu_test},
            "full": {"mIoU": m_full, "pixel_acc": acc_full, "per_class_IoU": iu_full}
        }, os.path.join(self.cmp_dir, "metrics_full_and_test.json"))

        plot_confusion(cm_te,   CLASS_NAMES, os.path.join(self.fig_dir, "confmat_test_counts.png"), normalize=False)
        plot_confusion(cm_te,   CLASS_NAMES, os.path.join(self.fig_dir, "confmat_test_norm.png"),   normalize=True)
        plot_confusion(cm_full, CLASS_NAMES, os.path.join(self.fig_dir, "confmat_full_counts.png"), normalize=False)
        plot_confusion(cm_full, CLASS_NAMES, os.path.join(self.fig_dir, "confmat_full_norm.png"),   normalize=True)
        plot_bars(iu_test, CLASS_NAMES, "IoU par classe (TEST)", "IoU", os.path.join(self.fig_dir, "per_class_IoU_test.png"))
        plot_bars(iu_full, CLASS_NAMES, "IoU par classe (FULL)", "IoU", os.path.join(self.fig_dir, "per_class_IoU_full.png"))
        gt_counts_test   = cm_te.sum(axis=1)
        pred_counts_test = cm_te.sum(axis=0)
        plot_double_bars(gt_counts_test, pred_counts_test, CLASS_NAMES,
                         "Distribution des classes (TEST)", "Pixels",
                         ["GT", "Pred"], os.path.join(self.fig_dir, "class_distribution_test.png"))

        print(f"[REPORT] TEST  mIoU={m_test:.4f} acc={acc_test:.4f}")
        print(f"[REPORT] FULL  mIoU={m_full:.4f} acc={acc_full:.4f}")
        print(f"[SAVED] Compare & RGB overlays -> {self.cmp_dir}")


# ======================================================================================
#                                   PART B : SSL + KMEANS
# ======================================================================================
class SSLExperiment:
    """
    Auto-encodeur (U-Net sortie 3 canaux) entraîné par reconstruction masquée.
    On extrait les features du bottleneck puis on fait un K-Means pour produire
    des cartes de clusters et des mosaïques (full/test).
    """
    def __init__(self, cfg, out_dir, device):
        self.cfg = cfg; self.out = out_dir; self.device = device
        self.Ltr, self.Lva, self.Lte, self.grid = make_loaders(cfg)

        feats = tuple(cfg.get("UNET_FEATS", [64,128,256,512]))
        self.net = UNet(in_ch=(4 if cfg.get("ADD_EXG", False) else 3), out_classes=3, feats=feats).to(device)

        self.opt = optim.AdamW(self.net.parameters(), lr=cfg.get("LR", 1e-3), weight_decay=cfg.get("WEIGHT_DECAY", 1e-4))
        self.epochs = cfg.get("EPOCHS", 100)
        self.mask_ratio = cfg.get("MASK_RATIO", 0.6)

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.ssl_dir, exist_ok=True)
        os.makedirs(self.cmp_dir, exist_ok=True)

    @property
    def ckpt_dir(self): return os.path.join(self.out, "checkpoints")
    @property
    def fig_dir(self):  return os.path.join(self.out, "figs")
    @property
    def ssl_dir(self):  return os.path.join(self.out, "ssl")
    @property
    def cmp_dir(self):  return os.path.join(self.out, "compare")

    def _random_mask(self, x):
        B, C, H, W = x.shape
        m = (torch.rand(B, 1, H, W, device=x.device) > self.mask_ratio).float()
        return x * m, m

    def run(self):
        # Phase 1: reconstruction SSL
        hist = {"train_recon": []}
        for ep in range(1, self.epochs + 1):
            self.net.train()
            loss_sum, n = 0.0, 0
            for x, _, _ in tqdm(self.Ltr, desc=f"SSL train {ep}/{self.epochs}", leave=False):
                x = x.to(self.device)
                xin, mask = self._random_mask(x)
                out, _ = self.net(xin, return_feats=True)
                loss = F.l1_loss(out * mask, x * mask) + 0.1 * F.mse_loss(out * mask, x * mask)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                b = x.size(0); n += b; loss_sum += loss.item() * b
            hist["train_recon"].append(loss_sum / max(n, 1))
            print(f"[SSL {ep}] recon={hist['train_recon'][-1]:.4f}")

        torch.save(self.net.state_dict(), os.path.join(self.ckpt_dir, "ssl_autoenc.pt"))

        # Phase 2: K-Means sur bottleneck (train)
        k = self.cfg.get("N_CLUSTERS", 6)
        feats_all = []
        self.net.eval()
        with torch.no_grad():
            for x, _, _ in tqdm(self.Ltr, desc="Encode train feats"):
                x = x.to(self.device)
                _, feats = self.net(x, return_feats=True)  # (B,C,H',W')
                B, C, Hp, Wp = feats.shape
                feats_all.append(feats.permute(0,2,3,1).reshape(-1, C).cpu().numpy())
        feats_all = np.concatenate(feats_all, axis=0)
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init="auto",
                                 random_state=self.cfg.get("SEED", 42))
        kmeans.fit(feats_all)

        # Phase 3: Projetter clusters → mosaïques + overlays RGB
        (n_rows, n_cols) = self.grid
        data_root = self.cfg.get("DATA_ROOT", "Dataset")

        def project_loader(loader):
            tiles = {}; names_all = []
            with torch.no_grad():
                for x, _, names in tqdm(loader, desc="Project clusters"):
                    x = x.to(self.device)
                    _, feats = self.net(x, return_feats=True)
                    B, C, Hp, Wp = feats.shape
                    v = feats.permute(0,2,3,1).reshape(-1, C).cpu().numpy()
                    labels = kmeans.predict(v).reshape(B, Hp, Wp)
                    labels_up = upsample_nearest(torch.from_numpy(labels).long(), (TILE_H, TILE_W)).cpu().numpy()
                    for i, nm in enumerate(names):
                        tiles[nm] = labels_up[i].astype(np.uint8)
                        names_all.append(nm)
            return tiles, names_all

        tr_tiles, tr_names = project_loader(self.Ltr)
        va_tiles, va_names = project_loader(self.Lva)
        te_tiles, te_names = project_loader(self.Lte)

        pred_all = {}; pred_all.update(tr_tiles); pred_all.update(va_tiles); pred_all.update(te_tiles)
        names_all = tr_names + va_names + te_names

        full_clusters = stitch_tiles(pred_all, n_rows, n_cols, TILE_H, TILE_W)
        test_clusters = crop_test_quadrant(full_clusters, n_rows, n_cols)

        # RGB + comparaisons
        rgb_tiles = load_rgb_tiles_by_names(data_root, names_all)
        full_rgb  = stitch_tiles_rgb(rgb_tiles, n_rows, n_cols, TILE_H, TILE_W)
        test_rgb  = crop_test_quadrant(full_rgb, n_rows, n_cols)

        save_png(full_rgb,                                  os.path.join(self.cmp_dir, "SSL_full_rgb.png"))
        save_png(test_rgb,                                  os.path.join(self.cmp_dir, "SSL_test_rgb.png"))
        save_png(colorize(full_clusters % CLASSES),         os.path.join(self.cmp_dir, "SSL_full_clusters.png"))
        save_png(colorize(test_clusters % CLASSES),         os.path.join(self.cmp_dir, "SSL_test_clusters.png"))
        save_png(concat_h(full_rgb, colorize(full_clusters % CLASSES)),
                 os.path.join(self.cmp_dir, "SSL_full_RGB_vs_CLUSTERS.png"))
        save_png(concat_h(test_rgb, colorize(test_clusters % CLASSES)),
                 os.path.join(self.cmp_dir, "SSL_test_RGB_vs_CLUSTERS.png"))

        # (Optionnel) évaluation non-supervisée sur val via appariement hongrois
        try:
            from scipy.optimize import linear_sum_assignment
            gt_all, pr_all = [], []
            with torch.no_grad():
                for x, y, _ in tqdm(self.Lva, desc="Hungarian on val"):
                    x = x.to(self.device); y = y.numpy()
                    _, feats = self.net(x, return_feats=True)
                    B, C, Hp, Wp = feats.shape
                    v = feats.permute(0,2,3,1).reshape(-1, C).cpu().numpy()
                    labels = kmeans.predict(v).reshape(B, Hp, Wp)
                    labels_up = upsample_nearest(torch.from_numpy(labels).long(), (TILE_H, TILE_W)).cpu().numpy()
                    gt_all.append(y); pr_all.append(labels_up)
            gt_all = np.concatenate(gt_all); pr_all = np.concatenate(pr_all)
            K = self.cfg.get("N_CLUSTERS", 6)
            cost = np.zeros((K, CLASSES))
            for k_id in range(K):
                for c in range(CLASSES):
                    inter = np.logical_and(pr_all == k_id, gt_all == c).sum()
                    union = np.logical_or(pr_all == k_id, gt_all == c).sum() + 1e-9
                    cost[k_id, c] = 1 - inter / union
            r, c = linear_sum_assignment(cost)
            mapping = {int(r[i]): int(c[i]) for i in range(len(r))}
            mapped = np.vectorize(lambda z: mapping.get(z, 0))(pr_all)
            miou, pixacc, percls = miou_score(gt_all, mapped, CLASSES)
            save_json({"unsup_val_mIoU": miou, "unsup_val_pixel_acc": pixacc,
                       "mapping": mapping, "per_class_IoU": percls},
                      os.path.join(self.ssl_dir, "unsup_val_metrics.json"))
            print(f"[SSL-VAL] mIoU (Hungarian)={miou:.4f} acc={pixacc:.4f}")
        except Exception as e:
            print(f"[SSL] Hungarian eval skipped: {e}")
