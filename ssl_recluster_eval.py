# ssl_recluster_eval.py
# Recluster SSL embeddings with any K (e.g., K=10) WITHOUT retraining.
# - Loads Results/<EXP>/ssl/best_ae.pt
# - Computes embeddings (latent features) from the encoder
# - Fits MiniBatchKMeans
# - Builds cluster->class mapping from TRAIN
# - Evaluates on VAL/TEST with confusion matrix + pixel acc + mIoU
#
# Usage example:
#   python ssl_recluster_eval.py --exp BestConfigB --k 10 --img_dir DataSet/images --mask_dir DataSet/masks
#
# If you already have saved embeddings, you can reuse them:
#   python ssl_recluster_eval.py --exp BestConfigB --k 10 --use_saved 1

import os, re, json, glob, math, argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception as e:
    raise RuntimeError("scikit-learn est requis (pip install scikit-learn).") from e


CLASS_NAMES = ["others", "water", "buildings", "farmlands", "green spaces"]
NUM_CLASSES = 5

PALETTE = np.array([
    [180, 180, 180],  # others
    [  0,  90, 200],  # water
    [200,   0,   0],  # buildings
    [230, 190,  30],  # farmlands
    [ 40, 160,  40],  # green spaces
], dtype=np.uint8)

_RC_RE   = re.compile(r".*r(\d+)[_-]c(\d+).*", re.IGNORECASE)          # ...r12_c34.png
_TILE_RE = re.compile(r".*tile[_-]?(\d+)[_-](\d+).*", re.IGNORECASE)   # tile_12_34.png


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def parse_rc(filename):
    base = os.path.basename(filename)
    m = _RC_RE.match(base)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _TILE_RE.match(base)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def list_files(root, exts=(".png",".jpg",".jpeg",".tif",".tiff")):
    root = str(root)
    return sorted([p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                   if os.path.isfile(p) and p.lower().endswith(exts)])

def pair_images_masks(img_dir, mask_dir):
    imgs = list_files(img_dir)
    pairs = []
    for ip in imgs:
        rel = os.path.relpath(ip, img_dir)
        mp = os.path.join(mask_dir, rel)
        if os.path.exists(mp):
            pairs.append((ip, mp))
    # fallback: match by basename if rel-structure differs
    if not pairs:
        masks = {os.path.splitext(os.path.basename(m))[0]: m for m in list_files(mask_dir)}
        for ip in imgs:
            k = os.path.splitext(os.path.basename(ip))[0]
            if k in masks:
                pairs.append((ip, masks[k]))
    return pairs

def split_pairs_quadrant(pairs, seed=42):
    """
    Same idea as your supervised code:
    - If filenames contain r/c -> test = lower-right quadrant
    - Else fallback random: 25% test, then val=20% of remaining
    """
    coords = [parse_rc(ip) for ip, _ in pairs]
    if any(r is None or c is None for r, c in coords):
        rng = np.random.RandomState(seed)
        idx = np.arange(len(pairs)); rng.shuffle(idx)
        n_test = int(0.25 * len(pairs))
        test_idx = idx[:n_test]
        rest = idx[n_test:]
        n_val = int(0.20 * len(rest))
        val_idx = rest[:n_val]
        train_idx = rest[n_val:]
        mode = "FALLBACK random"
    else:
        rows = np.array([r for r, _ in coords])
        cols = np.array([c for _, c in coords])
        r_thr = rows.max() * 0.5
        c_thr = cols.max() * 0.5
        test_idx = np.where((rows >= r_thr) & (cols >= c_thr))[0]
        rest_idx = np.setdiff1d(np.arange(len(pairs)), test_idx)
        rng = np.random.RandomState(seed); rng.shuffle(rest_idx)
        n_val = int(0.20 * len(rest_idx))
        val_idx = rest_idx[:n_val]
        train_idx = rest_idx[n_val:]
        mode = "QUADRANT lower-right"
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs   = [pairs[i] for i in val_idx]
    test_pairs  = [pairs[i] for i in test_idx]
    return train_pairs, val_pairs, test_pairs, mode

def mask_to_color(mask2d: np.ndarray):
    h, w = mask2d.shape
    return PALETTE[mask2d.reshape(-1)].reshape(h, w, 3)

def overlay(rgb_uint8: np.ndarray, mask_rgb_uint8: np.ndarray, alpha=0.45):
    rgb = rgb_uint8.astype(np.float32)
    msk = mask_rgb_uint8.astype(np.float32)
    out = (1.0 - alpha) * rgb + alpha * msk
    return np.clip(out, 0, 255).astype(np.uint8)

def plot_confusion_matrix(cm, out_path):
    fig = plt.figure(figsize=(7.5, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion matrix"); plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45)
    plt.yticks(ticks, CLASS_NAMES)
    thr = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{int(cm[i,j])}", ha="center",
                     color="white" if cm[i,j] > thr else "black")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + 1e-9)
    acc = tp.sum() / (cm.sum() + 1e-9)
    return {
        "pixel_acc": float(acc),
        "iou_per_class": [float(x) for x in iou],
        "miou": float(np.mean(iou)),
    }


class TileSegDataset(Dataset):
    def __init__(self, pairs, resize_hw=None):
        self.pairs = pairs
        self.resize_hw = resize_hw

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")
        if self.resize_hw is not None:
            H, W = self.resize_hw
            img = img.resize((W, H), Image.BILINEAR)
            msk = msk.resize((W, H), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0
        msk = np.asarray(msk, dtype=np.int64)
        img = torch.from_numpy(img).permute(2,0,1)  # CHW
        msk = torch.from_numpy(msk)                 # HW
        return img, msk


# ---------- Model loading (best effort) ----------

def try_import_autoencoder_class():
    candidates = [
        ("Networks.Architectures.ssl_autoencoder", ["AutoEncoder", "ConvAutoEncoder", "AE", "SSLAutoEncoder"]),
        ("Networks.Architectures.autoencoder",     ["AutoEncoder", "ConvAutoEncoder", "AE"]),
        ("Networks.Architectures.ae",              ["AutoEncoder", "ConvAutoEncoder", "AE"]),
        ("Networks.ssl_autoencoder",               ["AutoEncoder", "ConvAutoEncoder", "AE", "SSLAutoEncoder"]),
        ("Networks.ssl_model",                     ["AutoEncoder", "ConvAutoEncoder", "AE", "SSLAutoEncoder"]),
    ]
    for mod, names in candidates:
        try:
            m = __import__(mod, fromlist=["*"])
            for n in names:
                if hasattr(m, n):
                    cls = getattr(m, n)
                    if isinstance(cls, type) and issubclass(cls, nn.Module):
                        return cls
        except Exception:
            pass
    return None

class FallbackConvAE(nn.Module):
    """
    Fallback AE architecture ONLY if we cannot import yours.
    It may not match your checkpoint. We only use it to try load_state_dict(strict=False).
    """
    def __init__(self, in_ch=3, base=32, z=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base*2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*4, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*4, z, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z, base*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, in_ch, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state", "model", "net", "ae_state"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj if isinstance(obj, dict) else None

def matched_ratio(state_dict, model_state_dict):
    a = set(state_dict.keys())
    b = set(model_state_dict.keys())
    inter = len(a & b)
    return inter / max(1, len(a))

def load_ae_model(ckpt_path, device):
    obj = torch.load(ckpt_path, map_location="cpu")

    # Case 1: checkpoint is a full nn.Module
    if isinstance(obj, nn.Module):
        model = obj
        model.to(device).eval()
        return model

    # Case 2: checkpoint contains nn.Module
    if isinstance(obj, dict):
        for k in ["model", "net", "ae", "autoencoder"]:
            if k in obj and isinstance(obj[k], nn.Module):
                model = obj[k]
                model.to(device).eval()
                return model

    # Case 3: state_dict
    sd = unwrap_state_dict(obj)
    if sd is None:
        raise RuntimeError(f"Checkpoint {ckpt_path} illisible (pas module / pas state_dict).")

    cls = try_import_autoencoder_class()
    if cls is not None:
        # try common constructors
        for ctor in [
            lambda: cls(),
            lambda: cls(in_ch=3),
            lambda: cls(in_channels=3),
            lambda: cls(3),
        ]:
            try:
                model = ctor()
                r = matched_ratio(sd, model.state_dict())
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if r >= 0.6:
                    model.to(device).eval()
                    print(f"[OK] AE importée: {cls.__name__} | matched_keys≈{r:.2f} | missing={len(missing)} unexpected={len(unexpected)}")
                    return model
            except Exception:
                pass

    # fallback AE
    model = FallbackConvAE(in_ch=3)
    r = matched_ratio(sd, model.state_dict())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if r < 0.6:
        print(f"[WARN] Checkpoint ne matche pas bien une AE standard (matched_keys≈{r:.2f}).")
        print("       Si ça plante ensuite, il faut que ton AE expose `encoder` ou `encode()` et qu'on importe la bonne classe.")
    model.to(device).eval()
    return model

@torch.no_grad()
def encode_features(model, imgs):
    """
    Returns features as (B, C, Hf, Wf) or (B, D)
    We prefer encoder features.
    """
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        z = model.encode(imgs)
        return z
    if hasattr(model, "encoder"):
        z = model.encoder(imgs)
        return z
    # fallback: use penultimate if forward returns tuple
    out = model(imgs)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[1]
    # last resort: use avg pooled input (won't be great but avoids crash)
    return F.adaptive_avg_pool2d(imgs, (8, 8))


def sample_points(feats, masks=None, points_per_img=256, stratified=False, num_classes=NUM_CLASSES):
    """
    feats: (B,C,H,W) or (B,D)
    returns X: (N,D), y: (N,) optional
    """
    if feats.dim() == 2:
        X = feats
        y = None
        if masks is not None:
            # tile-level label = majority class in mask
            y = []
            for m in masks:
                v, c = torch.unique(m.view(-1), return_counts=True)
                y.append(int(v[c.argmax()].item()))
            y = torch.tensor(y, device=feats.device)
        return X, y

    B, C, H, W = feats.shape
    feats_flat = feats.permute(0,2,3,1).reshape(B, H*W, C)
    if masks is not None:
        masks_flat = masks.view(B, H*W)

    xs = []
    ys = []
    for b in range(B):
        n = min(points_per_img, H*W)
        if (masks is not None) and stratified:
            # sample roughly equally across classes present
            mb = masks_flat[b]
            idx_all = []
            per_c = max(1, n // num_classes)
            for c in range(num_classes):
                idx_c = torch.where(mb == c)[0]
                if idx_c.numel() == 0:
                    continue
                take = min(per_c, idx_c.numel())
                sel = idx_c[torch.randint(0, idx_c.numel(), (take,), device=mb.device)]
                idx_all.append(sel)
            if len(idx_all) == 0:
                idx = torch.randint(0, H*W, (n,), device=feats.device)
            else:
                idx = torch.cat(idx_all, dim=0)
                if idx.numel() < n:
                    extra = torch.randint(0, H*W, (n - idx.numel(),), device=feats.device)
                    idx = torch.cat([idx, extra], dim=0)
                else:
                    idx = idx[:n]
        else:
            idx = torch.randint(0, H*W, (n,), device=feats.device)

        xs.append(feats_flat[b, idx, :])
        if masks is not None:
            ys.append(masks_flat[b, idx])

    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0) if masks is not None else None
    return X, y


def build_cluster_mapping(kmeans, model, loader, device, k, mapping_mode="balanced",
                          points_per_img=256, stratified=True):
    """
    Build cluster->class mapping using TRAIN.
    mapping_mode:
      - majority: argmax count(cluster, class)
      - balanced: argmax count(cluster,class)/class_total   (helps minority classes)
    """
    counts = np.zeros((k, NUM_CLASSES), dtype=np.int64)
    class_tot = np.zeros((NUM_CLASSES,), dtype=np.int64)

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        feats = encode_features(model, imgs)
        X, y = sample_points(feats, masks=masks, points_per_img=points_per_img, stratified=stratified)

        Xn = X.detach().float().cpu().numpy()
        yn = y.detach().cpu().numpy()

        # predict clusters
        cl = kmeans.predict(Xn)  # (N,)
        for c in range(NUM_CLASSES):
            class_tot[c] += int((yn == c).sum())

        for ci in range(k):
            m = (cl == ci)
            if m.any():
                # bincount over classes for points in cluster
                bc = np.bincount(yn[m], minlength=NUM_CLASSES)
                counts[ci] += bc

    mapping = {}
    for ci in range(k):
        if mapping_mode == "balanced":
            score = counts[ci] / (class_tot + 1e-9)
            mapping[ci] = int(score.argmax())
        else:
            mapping[ci] = int(counts[ci].argmax())
    return mapping, counts, class_tot


@torch.no_grad()
def eval_with_mapping(kmeans, mapping, model, loader, device, k, out_dir, split="test"):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    # qualitative samples
    qual_dir = os.path.join(out_dir, f"quali_{split}_clusters")
    ensure_dir(qual_dir)
    saved = 0

    for bidx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)

        feats = encode_features(model, imgs)
        if feats.dim() == 2:
            # tile-level fallback
            Xn = feats.detach().float().cpu().numpy()
            cl = kmeans.predict(Xn)
            pred_cls = np.array([mapping[int(ci)] for ci in cl], dtype=np.int64)
            # no pixel-level cm possible here -> approximate by majority class per tile
            true_cls = []
            for m in masks:
                v, c = torch.unique(m.view(-1), return_counts=True)
                true_cls.append(int(v[c.argmax()].item()))
            true_cls = np.array(true_cls, dtype=np.int64)
            for t, p in zip(true_cls, pred_cls):
                cm[t, p] += 1
            continue

        B, C, H, W = feats.shape
        feats_flat = feats.permute(0,2,3,1).reshape(-1, C)
        masks_flat = masks.reshape(-1).detach().cpu().numpy()

        Xn = feats_flat.detach().float().cpu().numpy()
        cl = kmeans.predict(Xn)  # (B*H*W,)
        pred_flat = np.array([mapping[int(ci)] for ci in cl], dtype=np.int64)

        # update confusion matrix
        valid = (masks_flat >= 0) & (masks_flat < NUM_CLASSES)
        t = masks_flat[valid]
        p = pred_flat[valid]
        idx = NUM_CLASSES * t + p
        bc = np.bincount(idx, minlength=NUM_CLASSES*NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
        cm += bc

        # save a few qualitative tiles
        if saved < 10:
            # take first tile of the batch
            i = 0
            rgb = (imgs[i].detach().cpu().clamp(0,1).permute(1,2,0).numpy() * 255).astype(np.uint8)
            gt  = masks[i].detach().cpu().numpy().astype(np.int64)

            # predicted for this tile
            feats_i = feats[i].permute(1,2,0).reshape(-1, C).detach().cpu().numpy()
            cl_i = kmeans.predict(feats_i)
            pr_i = np.array([mapping[int(ci)] for ci in cl_i], dtype=np.int64).reshape(H, W)

            gt_rgb = mask_to_color(gt)
            pr_rgb = mask_to_color(pr_i)

            Image.fromarray(rgb).save(os.path.join(qual_dir, f"{saved:02d}_rgb.png"))
            Image.fromarray(gt_rgb).save(os.path.join(qual_dir, f"{saved:02d}_gt.png"))
            Image.fromarray(pr_rgb).save(os.path.join(qual_dir, f"{saved:02d}_pred.png"))
            Image.fromarray(overlay(rgb, pr_rgb)).save(os.path.join(qual_dir, f"{saved:02d}_overlay_pred.png"))
            saved += 1

    mets = metrics_from_cm(cm)
    with open(os.path.join(out_dir, f"metrics_{split}.json"), "w") as f:
        json.dump(mets, f, indent=2)
    plot_confusion_matrix(cm, os.path.join(out_dir, f"confusion_matrix_{split}.png"))
    return mets, cm


def try_load_yaml_config(exp):
    """
    Best effort: try typical paths in your project.
    """
    if yaml is None:
        return None, None
    candidates = [
        os.path.join("Todo_List", exp, "config.yaml"),
        os.path.join("Todo_List", "bestconfig", "config.yaml"),
        os.path.join("experiments", exp, "config.yaml"),
        os.path.join("experiments", "exp_best", "config.yaml"),
        os.path.join("Results", exp, "config.yaml"),
        os.path.join("Results", exp, "ssl", "config.yaml"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg, p
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, default="BestConfigB")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--use_saved", type=int, default=0, help="1=use cached embeddings if found")
    ap.add_argument("--img_dir", type=str, default=None)
    ap.add_argument("--mask_dir", type=str, default=None)
    ap.add_argument("--resize", type=str, default="64x64")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--points_per_img_fit", type=int, default=256)
    ap.add_argument("--points_per_img_map", type=int, default=256)
    ap.add_argument("--stratified", type=int, default=1)
    ap.add_argument("--mapping_mode", type=str, default="balanced", choices=["balanced","majority"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    exp_ssl_dir = os.path.join("Results", args.exp, "ssl")
    if not os.path.isdir(exp_ssl_dir):
        raise RuntimeError(f"Je ne trouve pas le dossier: {exp_ssl_dir}")

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    # config best effort (to find img_dir/mask_dir/resize)
    cfg, cfg_path = try_load_yaml_config(args.exp)
    if cfg_path:
        print(f"[Config] loaded: {cfg_path}")

    # image/mask dirs
    img_dir = args.img_dir
    mask_dir = args.mask_dir
    if (img_dir is None or mask_dir is None) and cfg is not None:
        # try typical keys
        def g(keys):
            cur = cfg
            for k in keys.split("."):
                if not isinstance(cur, dict) or (k not in cur):
                    return None
                cur = cur[k]
            return cur
        img_dir = img_dir or g("DATASET.IMG_DIR") or g("DATA.IMG_DIR") or g("PATH.IMG_DIR") or g("DIRS.IMG_DIR")
        mask_dir = mask_dir or g("DATASET.MASK_DIR") or g("DATA.MASK_DIR") or g("PATH.MASK_DIR") or g("DIRS.MASK_DIR")
        if isinstance(g("DATASET.RESIZE_SHAPE"), str):
            args.resize = g("DATASET.RESIZE_SHAPE")

    if img_dir is None or mask_dir is None:
        raise RuntimeError(
            "Je ne peux pas deviner img_dir/mask_dir.\n"
            "Relance comme ça:\n"
            "  python ssl_recluster_eval.py --exp BestConfigB --k 10 --img_dir <...> --mask_dir <...>\n"
        )

    # resize
    if isinstance(args.resize, str) and "x" in args.resize.lower():
        H, W = args.resize.lower().split("x")
        resize_hw = (int(H), int(W))
    else:
        resize_hw = None

    print(f"[Data] img_dir={img_dir}\n       mask_dir={mask_dir}\n       resize={resize_hw}")

    pairs = pair_images_masks(img_dir, mask_dir)
    if not pairs:
        raise RuntimeError("Aucun (image, mask) pair trouvé. Vérifie tes chemins.")

    train_pairs, val_pairs, test_pairs, mode = split_pairs_quadrant(pairs, seed=args.seed)
    print(f"[Split] mode={mode} | train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    train_ds = TileSegDataset(train_pairs, resize_hw=resize_hw)
    val_ds   = TileSegDataset(val_pairs,   resize_hw=resize_hw)
    test_ds  = TileSegDataset(test_pairs,  resize_hw=resize_hw)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))

    # checkpoint path
    ckpt_candidates = [
        os.path.join(exp_ssl_dir, "best_ae.pt"),
        os.path.join(exp_ssl_dir, "best_model.pt"),
        os.path.join(exp_ssl_dir, "ae_best.pt"),
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        raise RuntimeError(f"Je ne trouve pas best_ae.pt (ou best_model.pt) dans {exp_ssl_dir}")

    print(f"[CKPT] {ckpt_path}")
    model = load_ae_model(ckpt_path, device=device)

    out_dir = os.path.join(exp_ssl_dir, f"recluster_k{args.k}")
    ensure_dir(out_dir)

    # ---- caching embeddings (optional but helpful)
    # We'll cache per-split sampled embeddings (for fast reruns). FULL pixel embeddings would be huge.
    cache_train = os.path.join(out_dir, "emb_train_sampled.npz")
    cache_val   = os.path.join(out_dir, "emb_val_sampled.npz")
    cache_test  = os.path.join(out_dir, "emb_test_sampled.npz")

    # ---- Fit KMeans on TRAIN
    print(f"[KMeans] fitting MiniBatchKMeans(k={args.k}) on TRAIN sampled points...")
    kmeans = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, batch_size=8192, n_init="auto")

    if args.use_saved == 1 and os.path.exists(cache_train):
        dat = np.load(cache_train)
        X = dat["X"]
        kmeans.fit(X)
        print(f"[KMeans] fit from cache: {cache_train} | X={X.shape}")
    else:
        X_all = []
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            feats = encode_features(model, imgs)
            X, _ = sample_points(
                feats, masks=masks,
                points_per_img=args.points_per_img_fit,
                stratified=bool(args.stratified),
                num_classes=NUM_CLASSES
            )
            Xn = X.detach().float().cpu().numpy()
            kmeans.partial_fit(Xn)
            X_all.append(Xn)
        # save sampled embeddings for faster future
        Xcat = np.concatenate(X_all, axis=0) if len(X_all) else np.zeros((0, 1), dtype=np.float32)
        np.savez_compressed(cache_train, X=Xcat)
        print(f"[Cache] saved: {cache_train} | X={Xcat.shape}")

    # ---- Build cluster->class mapping from TRAIN (uses GT only for mapping, not for training AE)
    print(f"[Mapping] building cluster->class mapping on TRAIN ({args.mapping_mode})...")
    mapping, counts, class_tot = build_cluster_mapping(
        kmeans, model, train_loader, device=device, k=args.k,
        mapping_mode=args.mapping_mode,
        points_per_img=args.points_per_img_map,
        stratified=bool(args.stratified)
    )
    with open(os.path.join(out_dir, "cluster_mapping.json"), "w") as f:
        json.dump({
            "k": args.k,
            "mapping_mode": args.mapping_mode,
            "mapping": {str(k): int(v) for k, v in mapping.items()},
            "counts_cluster_x_class": counts.tolist(),
            "class_totals_used_for_mapping": class_tot.tolist(),
        }, f, indent=2)
    print(f"[Mapping] saved: {os.path.join(out_dir, 'cluster_mapping.json')}")

    # ---- Evaluate VAL/TEST
    print("[Eval] VAL...")
    mets_val, cm_val = eval_with_mapping(kmeans, mapping, model, val_loader, device, args.k, out_dir, split="val")
    print(f"[VAL] pixel_acc={mets_val['pixel_acc']:.3f} | mIoU={mets_val['miou']:.3f} | IoU={mets_val['iou_per_class']}")

    print("[Eval] TEST...")
    mets_test, cm_test = eval_with_mapping(kmeans, mapping, model, test_loader, device, args.k, out_dir, split="test")
    print(f"[TEST] pixel_acc={mets_test['pixel_acc']:.3f} | mIoU={mets_test['miou']:.3f} | IoU={mets_test['iou_per_class']}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"val": mets_val, "test": mets_test}, f, indent=2)

    print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
