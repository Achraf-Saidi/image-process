# Networks/model.py
import os, json, math, glob, re, time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torch.cuda import amp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from Networks.Architectures.unet_resse import UNetResSE
except Exception:
    from .Architectures.unet_resse import UNetResSE

CLASS_NAMES = ["others", "water", "buildings", "farmlands", "green spaces"]

def _ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def _get(cfg, path, default=None):
    cur = cfg
    for k in path.split('.'):
        if not isinstance(cur, dict): return default
        cur = cur.get(k, None)
    return default if cur is None else cur

def _list_files(root, exts=(".png",".jpg",".jpeg",".tif",".tiff")):
    return sorted([p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                   if os.path.isfile(p) and p.lower().endswith(exts)])

def _pair_images_masks(img_dir, msk_dir):
    imgs = _list_files(img_dir)
    pairs = []
    for ip in imgs:
        rel = os.path.relpath(ip, img_dir)
        mp = os.path.join(msk_dir, rel)
        if os.path.exists(mp):
            pairs.append((ip, mp))
    return pairs

_RC_RE   = re.compile(r".*r(\d+)[_-]c(\d+).*", re.IGNORECASE)
_TILE_RE = re.compile(r".*tile[_-]?(\d+)[_-](\d+).*", re.IGNORECASE)

def _parse_rc(filename):
    base = os.path.basename(filename)
    m = _RC_RE.match(base)
    if m: return int(m.group(1)), int(m.group(2))
    m = _TILE_RE.match(base)
    if m: return int(m.group(1)), int(m.group(2))
    return None, None

def _mask_to_color(mask):
    palette = np.array([
        [180, 180, 180],  # others
        [  0,  90, 200],  # water
        [200,   0,   0],  # buildings
        [230, 190,  30],  # farmlands
        [ 40, 160,  40],  # green spaces
    ], dtype=np.uint8)
    h, w = mask.shape
    mask = np.clip(mask, 0, len(palette)-1)
    return palette[mask.reshape(-1)].reshape(h, w, 3)

def _plot_curves(train_vals, val_vals, ylabel, out_path):
    epochs = range(1, len(train_vals)+1)
    plt.figure()
    plt.plot(epochs, train_vals, label='train')
    plt.plot(epochs, val_vals, label='val')
    plt.xlabel('epoch'); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()

def _confusion_matrix(pred, target, n_cls):
    # pred/target: (H,W) int
    p = pred.reshape(-1)
    t = target.reshape(-1)
    mask = (t >= 0) & (t < n_cls)
    idx = n_cls * t[mask] + p[mask]
    cm = np.bincount(idx, minlength=n_cls*n_cls).reshape(n_cls, n_cls)
    return cm

def _metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou  = tp / (tp + fp + fn + 1e-6)
    dice = 2*tp / (2*tp + fp + fn + 1e-6)
    acc  = tp.sum() / (cm.sum() + 1e-6)
    return {
        "pixel_acc": float(acc),
        "miou": float(iou.mean()),
        "mdice": float(dice.mean()),
        "iou_per_class": [float(x) for x in iou],
        "dice_per_class": [float(x) for x in dice],
    }

class TileDataset(Dataset):
    def __init__(self, pairs, resize_hw=(64,64), augment=False):
        self.pairs = pairs
        self.resize_hw = resize_hw
        self.augment = augment

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert("RGB").resize((self.resize_hw[1], self.resize_hw[0]), Image.BILINEAR)
        msk = Image.open(mp).convert("L").resize((self.resize_hw[1], self.resize_hw[0]), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0
        msk = np.asarray(msk, dtype=np.int64)
        img = torch.from_numpy(img).permute(2,0,1)  # CHW
        msk = torch.from_numpy(msk)                 # HW
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2]); msk = torch.flip(msk, dims=[1])
            if torch.rand(1).item() < 0.25:
                k = int(torch.randint(0,4,(1,)).item())
                if k:
                    img = torch.rot90(img, k, dims=[1,2])
                    msk = torch.rot90(msk, k, dims=[0,1])
        return img, msk

class ShowTileDataset(Dataset):
    # For showDataset(): returns 4 values
    def __init__(self, base_ds):
        self.base = base_ds
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, mask = self.base[idx]
        ip, _ = self.base.pairs[idx]
        tileName = os.path.splitext(os.path.basename(ip))[0]
        resizedImg = (img * 255.0).round().clamp(0,255).to(torch.uint8)
        return img, mask, tileName, resizedImg

class Network_Class:
    def __init__(self, param, img_dir, mask_dir, results_path):
        self.cfg = param or {}
        self.results_dir = results_path
        _ensure_dir(self.results_dir)

        self.mode = str(self.cfg.get("MODE", "segmentation")).lower()

        want_cuda = str(self.cfg.get("DEVICE", "cpu")).lower() == "cuda"
        self.device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")
        self.use_amp = (self.device.type == "cuda")
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        if self.device.type == "cuda":
            cudnn.benchmark = True
            props = torch.cuda.get_device_properties(0)
            print(f"[CUDA] {torch.cuda.get_device_name(0)} | {props.total_memory/1024**3:.1f} GB")

        # data
        self.num_classes = int(_get(self.cfg, "DATA.NUM_CLASSES", 5))
        resize = str(_get(self.cfg, "DATASET.RESIZE_SHAPE", "64x64")).lower()
        h,w = resize.split("x")
        self.resize_hw = (int(h), int(w))

        pairs = _pair_images_masks(img_dir, mask_dir)
        if not pairs:
            raise RuntimeError("No (image, mask) pairs found.")

        # --- split by quadrant if filenames have indices, else fallback random
        coords = [(_parse_rc(ip)[0], _parse_rc(ip)[1]) for ip,_ in pairs]
        if any(r is None for r,_ in coords) or any(c is None for _,c in coords):
            rng = np.random.RandomState(42)
            idx = np.arange(len(pairs)); rng.shuffle(idx)
            n_test = int(0.25*len(pairs))
            test_idx = idx[:n_test]
            rest = idx[n_test:]
            n_val = int(0.20*len(rest))
            val_idx = rest[:n_val]
            train_idx = rest[n_val:]
            split_mode = "FALLBACK (random)"
        else:
            rows = np.array([r for r,_ in coords]); cols = np.array([c for _,c in coords])
            r_thr = rows.max()*0.5; c_thr = cols.max()*0.5
            test_idx = np.where((rows >= r_thr) & (cols >= c_thr))[0]
            rest_idx = np.setdiff1d(np.arange(len(pairs)), test_idx)
            rng = np.random.RandomState(42); rng.shuffle(rest_idx)
            n_val = int(0.20*len(rest_idx))
            val_idx = rest_idx[:n_val]
            train_idx = rest_idx[n_val:]
            split_mode = "QUADRANT (lower-right)"

        self.train_pairs = [pairs[i] for i in train_idx]
        self.val_pairs   = [pairs[i] for i in val_idx]
        self.test_pairs  = [pairs[i] for i in test_idx]
        print(f"[Split] mode={split_mode} | train={len(self.train_pairs)} val={len(self.val_pairs)} test={len(self.test_pairs)}")

        bs = int(_get(self.cfg, "TRAIN.BATCH_SIZE", 64))
        nw = int(_get(self.cfg, "TRAIN.NUM_WORKERS", 4))
        pin = (self.device.type == "cuda")

        self.ds_train = TileDataset(self.train_pairs, resize_hw=self.resize_hw, augment=True)
        self.ds_val   = TileDataset(self.val_pairs,   resize_hw=self.resize_hw, augment=False)
        self.ds_test  = TileDataset(self.test_pairs,  resize_hw=self.resize_hw, augment=False)
        self.dataSetTrain = ShowTileDataset(self.ds_train)

        self.train_loader = DataLoader(self.ds_train, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=pin)
        self.val_loader   = DataLoader(self.ds_val,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
        self.test_loader  = DataLoader(self.ds_test,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)

        # model
        in_ch   = int(_get(self.cfg, "MODEL.IN_CHANNELS", 3))
        base_ch = int(_get(self.cfg, "MODEL.BASE_CH", 48))
        drop    = float(_get(self.cfg, "MODEL.DROPOUT", 0.10))
        proj_dim = int(_get(self.cfg, "SSL.PROJ_DIM", 64))

        if self.mode == "ssl":
            self.net = UNetResSE(in_ch=in_ch, out_ch=3, base_ch=base_ch, dropout=drop, task="recon", proj_dim=proj_dim).to(self.device)
        else:
            self.net = UNetResSE(in_ch=in_ch, out_ch=self.num_classes, base_ch=base_ch, dropout=drop, task="seg", proj_dim=proj_dim).to(self.device)

        # optimizer & training
        lr = float(_get(self.cfg, "TRAIN.LR", 1e-3))
        wd = float(_get(self.cfg, "TRAIN.WEIGHT_DECAY", 1e-4))
        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)

        self.epochs   = int(_get(self.cfg, "TRAIN.EPOCHS", 60))
        self.patience = int(_get(self.cfg, "TRAIN.PATIENCE", 10))

        # losses
        if self.mode == "ssl":
            self.l1_w  = float(_get(self.cfg, "SSL.L1_W", 1.0))
            self.mse_w = float(_get(self.cfg, "SSL.MSE_W", 0.2))
            self.best_is_min = True
        else:
            self.ce = nn.CrossEntropyLoss()
            self.best_is_min = False

        # ssl corruption params
        self.mask_ratio = float(_get(self.cfg, "SSL.MASK_RATIO", 0.6))
        self.mask_patch = int(_get(self.cfg, "SSL.MASK_PATCH", 8))
        self.noise_std  = float(_get(self.cfg, "SSL.NOISE_STD", 0.03))
        self.feature_level = str(_get(self.cfg, "SSL.FEATURE_LEVEL", "enc2")).lower()
        self.k_clusters = int(_get(self.cfg, "SSL.CLUSTERS_K", 10))

        self.best_path = os.path.join(self.results_dir, "best_model.pt")

        print(f"[Mode] {self.mode} | [Device] {self.device} | bs={bs} epochs={self.epochs}")

    def _corrupt(self, x):
        """
        Masked reconstruction proxy:
        - random patch mask (ratio)
        - masked pixels replaced by 0 + noise
        """
        B,C,H,W = x.shape
        ps = self.mask_patch
        gh, gw = H//ps, W//ps
        # grid mask (B,gh,gw)
        m = (torch.rand(B, gh, gw, device=x.device) < self.mask_ratio).float()
        m = m.repeat_interleave(ps, 1).repeat_interleave(ps, 2)  # (B,H,W)
        m = m.unsqueeze(1)  # (B,1,H,W)
        noise = torch.randn_like(x) * self.noise_std
        x_cor = x * (1 - m) + (0.0 + noise) * m
        return x_cor.clamp(0,1)

    def _save_ssl_quali(self, epoch_tag="ssl", max_samples=12):
        out_path = os.path.join(self.results_dir, f"ssl_recon_{epoch_tag}.png")
        self.net.eval()
        tiles = []
        with torch.no_grad():
            for imgs, _ in self.val_loader:
                imgs = imgs.to(self.device)
                cor  = self._corrupt(imgs)
                with amp.autocast(enabled=self.use_amp):
                    rec = self.net(cor)
                imgs = imgs.detach().cpu()
                cor  = cor.detach().cpu()
                rec  = rec.detach().cpu()
                for i in range(min(imgs.size(0), max_samples//3)):
                    tiles.append(cor[i])
                    tiles.append(rec[i])
                    tiles.append(imgs[i])
                break
        if not tiles:
            return
        # make grid: 3 columns (cor, rec, gt)
        n = len(tiles)
        ncol = 3
        nrow = int(math.ceil(n/ncol))
        C,H,W = tiles[0].shape
        pad = 2
        grid = torch.ones((H*nrow + pad*(nrow-1), W*ncol + pad*(ncol-1), 3), dtype=torch.float32)
        idx = 0
        for r in range(nrow):
            for c in range(ncol):
                if idx >= n: break
                img = tiles[idx].clamp(0,1).permute(1,2,0)
                y = r*(H+pad); x = c*(W+pad)
                grid[y:y+H, x:x+W, :] = img
                idx += 1
        plt.figure(figsize=(10, 2.5*nrow))
        plt.imshow(grid.numpy()); plt.axis("off")
        plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

    def train(self):
        if os.path.exists(self.best_path):
            print(f"[Info] best_model.pt already exists -> training will overwrite it.")

        best_score = float("inf") if self.best_is_min else -1e9
        no_improve = 0
        tr_hist, va_hist = [], []

        for ep in range(1, self.epochs+1):
            t0 = time.time()
            # ---- TRAIN
            self.net.train()
            tr_loss = 0.0
            n_tr = 0
            for imgs, masks in self.train_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                if self.mode == "ssl":
                    cor = self._corrupt(imgs)
                    with amp.autocast(enabled=self.use_amp):
                        rec = self.net(cor)
                        l1  = F.l1_loss(rec, imgs)
                        mse = F.mse_loss(rec, imgs)
                        loss = self.l1_w*l1 + self.mse_w*mse
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    with amp.autocast(enabled=self.use_amp):
                        logits = self.net(imgs)
                        loss = self.ce(logits.float(), masks.long())
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                tr_loss += loss.item() * imgs.size(0)
                n_tr += imgs.size(0)

            tr_loss /= max(n_tr, 1)

            # ---- VAL
            self.net.eval()
            va_loss = 0.0
            n_va = 0
            with torch.no_grad():
                for imgs, masks in self.val_loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)
                    if self.mode == "ssl":
                        cor = self._corrupt(imgs)
                        with amp.autocast(enabled=self.use_amp):
                            rec = self.net(cor)
                            l1  = F.l1_loss(rec, imgs)
                            mse = F.mse_loss(rec, imgs)
                            loss = self.l1_w*l1 + self.mse_w*mse
                    else:
                        with amp.autocast(enabled=self.use_amp):
                            logits = self.net(imgs)
                            loss = self.ce(logits.float(), masks.long())
                    va_loss += loss.item() * imgs.size(0)
                    n_va += imgs.size(0)
            va_loss /= max(n_va, 1)

            tr_hist.append(tr_loss); va_hist.append(va_loss)
            _plot_curves(tr_hist, va_hist, "loss", os.path.join(self.results_dir, "curves_loss.png"))

            # model selection
            improved = False
            if self.best_is_min:
                score = va_loss
                if score < best_score - 1e-9:
                    best_score = score; improved = True
            else:
                # in segmentation, we can still pick best on val_loss (simple)
                score = -va_loss
                if score > best_score + 1e-9:
                    best_score = score; improved = True

            if improved:
                torch.save(self.net.state_dict(), self.best_path)
                no_improve = 0
            else:
                no_improve += 1

            dt = time.time() - t0
            print(f"[{ep}/{self.epochs}] train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | best={best_score:.4f} {'<<' if improved else ''} | no_improve={no_improve} | {dt:.1f}s")

            if self.mode == "ssl" and (ep == 1 or ep % 5 == 0):
                self._save_ssl_quali(epoch_tag=f"ep{ep}", max_samples=12)

            if no_improve >= self.patience:
                print(f"Early stop (no improvement for {self.patience} epochs).")
                break

        # load best
        if os.path.exists(self.best_path):
            self.net.load_state_dict(torch.load(self.best_path, map_location=self.device), strict=True)
            self._save_ssl_quali(epoch_tag="best", max_samples=12)

    def loadWeights(self):
        if os.path.exists(self.best_path):
            self.net.load_state_dict(torch.load(self.best_path, map_location=self.device), strict=True)

    def _collect_bounds(self, pairs):
        rc2pair = {}
        rows, cols = [], []
        for ip, mp in pairs:
            r, c = _parse_rc(ip)
            if r is None or c is None:
                return None, None
            rc2pair[(r, c)] = (ip, mp)
            rows.append(r); cols.append(c)
        return rc2pair, (min(rows), max(rows), min(cols), max(cols))

    def _overlay(self, rgb_uint8, mask_rgb_uint8, alpha=0.45):
        rgb = rgb_uint8.astype(np.float32)
        msk = mask_rgb_uint8.astype(np.float32)
        out = (1-alpha)*rgb + alpha*msk
        return np.clip(out, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def _extract_features_for_kmeans(self, pairs, max_samples=250000):
        feats = []
        self.net.eval()
        loader = DataLoader(TileDataset(pairs, resize_hw=self.resize_hw, augment=False),
                            batch_size=64, shuffle=False, num_workers=0)
        for imgs, _ in loader:
            imgs = imgs.to(self.device)
            with amp.autocast(enabled=self.use_amp):
                f = self.net.encode(imgs, level=self.feature_level, project=True)  # (B,C,H',W')
            f = f.float()
            f = F.normalize(f, dim=1)  # normalize channels
            B,C,Hf,Wf = f.shape
            f = f.permute(0,2,3,1).reshape(-1, C).detach().cpu().numpy()  # (N,C)
            feats.append(f)
            if sum(x.shape[0] for x in feats) >= max_samples:
                break
        X = np.concatenate(feats, axis=0)
        if X.shape[0] > max_samples:
            idx = np.random.RandomState(42).choice(X.shape[0], size=max_samples, replace=False)
            X = X[idx]
        return X

    @torch.no_grad()
    def _predict_clusters_tile(self, img_tensor, kmeans):
        """
        img_tensor: (1,3,64,64) on device
        returns: cluster map upscaled to (64,64) as np.int64
        """
        with amp.autocast(enabled=self.use_amp):
            f = self.net.encode(img_tensor, level=self.feature_level, project=True)  # (1,C,Hf,Wf)
        f = F.normalize(f.float(), dim=1)
        C,Hf,Wf = f.shape[1], f.shape[2], f.shape[3]
        X = f.permute(0,2,3,1).reshape(-1, C).detach().cpu().numpy()
        lab = kmeans.predict(X).reshape(Hf, Wf).astype(np.int64)

        # upscale to 64x64
        lab_t = torch.from_numpy(lab).unsqueeze(0).unsqueeze(0).float()  # 1,1,Hf,Wf
        lab_up = F.interpolate(lab_t, size=self.resize_hw, mode="nearest").squeeze().long().numpy()
        return lab_up

    def evaluate(self):
        self.loadWeights()

        if self.mode != "ssl":
            # --- simple supervised eval on test
            self.net.eval()
            cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            loss_sum, n = 0.0, 0
            ce = nn.CrossEntropyLoss()
            with torch.no_grad():
                for imgs, masks in self.test_loader:
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)
                    with amp.autocast(enabled=self.use_amp):
                        logits = self.net(imgs)
                        loss = ce(logits.float(), masks.long())
                    loss_sum += loss.item() * imgs.size(0)
                    n += imgs.size(0)
                    pred = logits.argmax(dim=1).detach().cpu().numpy()
                    gt   = masks.detach().cpu().numpy()
                    for i in range(pred.shape[0]):
                        cm += _confusion_matrix(pred[i], gt[i], self.num_classes)
            mets = _metrics_from_cm(cm)
            out_dir = os.path.join(self.results_dir, "eval_test")
            _ensure_dir(out_dir)
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump({**mets, "loss": float(loss_sum/max(n,1))}, f, indent=2)
            print(f"[TEST] loss={loss_sum/max(n,1):.4f} | acc={mets['pixel_acc']:.3f} | mIoU={mets['miou']:.3f}")
            return mets

        # --- SSL Part B: clustering + pseudo segmentation maps + comparison
        out_dir = os.path.join(self.results_dir, "ssl_eval")
        _ensure_dir(out_dir)

        # 1) fit KMeans on TRAIN features
        try:
            from sklearn.cluster import MiniBatchKMeans
        except Exception as e:
            raise RuntimeError("scikit-learn is required for clustering. Install: pip install scikit-learn") from e

        max_samp = int(_get(self.cfg, "SSL.CLUSTER_MAX_SAMPLES", 250000))
        X = self._extract_features_for_kmeans(self.train_pairs, max_samples=max_samp)

        kmeans_path = os.path.join(out_dir, f"kmeans_K{self.k_clusters}.joblib")
        # fit fresh each time (simple & reproducible)
        kmeans = MiniBatchKMeans(n_clusters=self.k_clusters, random_state=42, batch_size=20000, n_init="auto")
        kmeans.fit(X)

        # 2) compute cluster->class mapping using VAL (majority vote per cluster)
        counts = np.zeros((self.k_clusters, self.num_classes), dtype=np.int64)
        val_loader = DataLoader(TileDataset(self.val_pairs, resize_hw=self.resize_hw, augment=False),
                                batch_size=1, shuffle=False, num_workers=0)
        for img, gt in val_loader:
            img = img.to(self.device)
            gt = gt.squeeze(0).numpy().astype(np.int64)
            cl = self._predict_clusters_tile(img, kmeans)  # (64,64)
            for k in range(self.k_clusters):
                m = (cl == k)
                if m.any():
                    binc = np.bincount(gt[m].reshape(-1), minlength=self.num_classes)
                    counts[k] += binc

        mapping = counts.argmax(axis=1).tolist()  # cluster k -> class id
        with open(os.path.join(out_dir, f"cluster_to_class_K{self.k_clusters}.json"), "w") as f:
            json.dump({"mapping": mapping, "counts": counts.tolist()}, f, indent=2)

        # 3) build mosaics on TEST quadrant (clusters + mapped classes)
        rc = self._collect_bounds(self.test_pairs)
        if rc[0] is None:
            print("[SSL] Filenames have no tile indices -> cannot stitch mosaic. Saving only per-tile metrics.")
            return {}

        rc2pair, (rmin, rmax, cmin, cmax) = rc
        H = (rmax - rmin + 1) * self.resize_hw[0]
        W = (cmax - cmin + 1) * self.resize_hw[1]

        mosaic_rgb   = np.zeros((H, W, 3), dtype=np.uint8)
        mosaic_gt    = np.zeros((H, W), dtype=np.int64)
        mosaic_cl    = np.zeros((H, W), dtype=np.int64)
        mosaic_mapped= np.zeros((H, W), dtype=np.int64)

        for (r,c), (ip, mp) in rc2pair.items():
            y0 = (r - rmin) * self.resize_hw[0]
            x0 = (c - cmin) * self.resize_hw[1]

            rgb = Image.open(ip).convert("RGB").resize((self.resize_hw[1], self.resize_hw[0]), Image.BILINEAR)
            gt  = Image.open(mp).convert("L").resize((self.resize_hw[1], self.resize_hw[0]), Image.NEAREST)

            rgb = np.asarray(rgb, dtype=np.uint8)
            gt  = np.asarray(gt, dtype=np.int64)

            img_t = torch.from_numpy(rgb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(self.device)
            cl = self._predict_clusters_tile(img_t, kmeans)
            mapped = np.vectorize(lambda kk: mapping[kk])(cl).astype(np.int64)

            mosaic_rgb[y0:y0+self.resize_hw[0], x0:x0+self.resize_hw[1], :] = rgb
            mosaic_gt[y0:y0+self.resize_hw[0], x0:x0+self.resize_hw[1]] = gt
            mosaic_cl[y0:y0+self.resize_hw[0], x0:x0+self.resize_hw[1]] = cl
            mosaic_mapped[y0:y0+self.resize_hw[0], x0:x0+self.resize_hw[1]] = mapped

        # palettes
        rng = np.random.RandomState(0)
        cl_palette = rng.randint(0,255,(self.k_clusters,3),dtype=np.uint8)
        cl_palette[0] = np.array([20,20,20],dtype=np.uint8)

        cl_rgb = cl_palette[np.clip(mosaic_cl,0,self.k_clusters-1)]
        gt_rgb = _mask_to_color(mosaic_gt)
        mp_rgb = _mask_to_color(mosaic_mapped)

        Image.fromarray(mosaic_rgb).save(os.path.join(out_dir, "test_mosaic_rgb.png"))
        Image.fromarray(gt_rgb).save(os.path.join(out_dir, "test_mosaic_gt.png"))
        Image.fromarray(cl_rgb).save(os.path.join(out_dir, f"test_mosaic_clusters_K{self.k_clusters}.png"))
        Image.fromarray(mp_rgb).save(os.path.join(out_dir, f"test_mosaic_mapped_K{self.k_clusters}.png"))

        Image.fromarray(self._overlay(mosaic_rgb, cl_rgb)).save(os.path.join(out_dir, f"test_overlay_clusters_K{self.k_clusters}.png"))
        Image.fromarray(self._overlay(mosaic_rgb, mp_rgb)).save(os.path.join(out_dir, f"test_overlay_mapped_K{self.k_clusters}.png"))

        # 4) quantitative comparison vs GT (on test quadrant)
        cm = _confusion_matrix(mosaic_mapped, mosaic_gt, self.num_classes)
        mets = _metrics_from_cm(cm)
        with open(os.path.join(out_dir, f"metrics_K{self.k_clusters}.json"), "w") as f:
            json.dump({**mets, "K": self.k_clusters}, f, indent=2)

        print(f"[SSL TEST] K={self.k_clusters} | acc={mets['pixel_acc']:.3f} | mIoU(mapped)={mets['miou']:.3f} | mDice(mapped)={mets['mdice']:.3f}")
        return mets
