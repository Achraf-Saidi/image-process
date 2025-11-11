import os, re, json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

# ---------------- Tile helpers ----------------

TILE_RE = re.compile(r"tile_(\d+)_(\d+)\.png")

def parse_rr_cc(name):
    m = TILE_RE.match(name)
    if not m: raise ValueError(f"Bad tile name: {name}")
    return int(m.group(1)), int(m.group(2))

def discover_grid(all_filenames):
    rr_max, cc_max = 0, 0
    for fn in all_filenames:
        rr, cc = parse_rr_cc(os.path.basename(fn))
        rr_max = max(rr_max, rr); cc_max = max(cc_max, cc)
    return rr_max+1, cc_max+1  # n_rows, n_cols

def is_test_tile(rr, cc, n_rows, n_cols):
    # Lower-right quadrant = test (rows >= n_rows/2 and cols >= n_cols/2)
    return (rr >= n_rows//2) and (cc >= n_cols//2)

# ---------------- Colors / viz ----------------

PALETTE_5 = np.array([
    [ 60,  60,  60],   # 0 others -> dark gray
    [ 40, 140, 240],   # 1 water -> blue
    [220,  70,  70],   # 2 buildings -> red
    [240, 200,  60],   # 3 farmlands -> yellow
    [ 60, 180,  90],   # 4 green -> green
], dtype=np.uint8)

def colorize(mask_np, palette=PALETTE_5):
    return palette[mask_np.clip(0, len(palette)-1)]

def save_png(arr, path):
    Image.fromarray(arr).save(path)

# ---------------- Stitch (labels & RGB) ----------------

def stitch_tiles(tile_dict, n_rows, n_cols, tile_h=64, tile_w=64):
    """tile_dict: { 'tile_RR_CC.png': label_64x64_uint8 } -> (H,W) uint8"""
    H, W = n_rows*tile_h, n_cols*tile_w
    out = np.zeros((H, W), dtype=np.uint8)
    for name, mask in tile_dict.items():
        rr, cc = parse_rr_cc(name)
        r0, c0 = rr*tile_h, cc*tile_w
        out[r0:r0+tile_h, c0:c0+tile_w] = mask.astype(np.uint8)
    return out

def stitch_tiles_rgb(tile_dict, n_rows, n_cols, tile_h=64, tile_w=64):
    """tile_dict: { 'tile_RR_CC.png': rgb_64x64x3_uint8 } -> (H,W,3) uint8"""
    H, W = n_rows*tile_h, n_cols*tile_w
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for name, rgb in tile_dict.items():
        rr, cc = parse_rr_cc(name)
        r0, c0 = rr*tile_h, cc*tile_w
        out[r0:r0+tile_h, c0:c0+tile_w] = rgb.astype(np.uint8)
    return out

def overlay_segmentation(rgb_uint8, label_uint8, palette=PALETTE_5, alpha=0.45):
    """Retourne un overlay RGB: (1-alpha)*rgb + alpha*colorize(label)"""
    seg_rgb = colorize(label_uint8, palette).astype(np.float32)
    base = rgb_uint8.astype(np.float32)
    over = (1.0 - alpha) * base + alpha * seg_rgb
    return np.clip(over, 0, 255).astype(np.uint8)

# ---------------- Metrics ----------------

def fast_hist(a, b, n_class):
    k = (a >= 0) & (a < n_class)
    return np.bincount(n_class * a[k].astype(int) + b[k].astype(int), minlength=n_class**2).reshape(n_class, n_class)

def miou_score(y_true, y_pred, n_class):
    hist = fast_hist(y_true.flatten(), y_pred.flatten(), n_class)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    miou = np.nanmean(iu)
    pixacc = np.diag(hist).sum() / (hist.sum() + 1e-10)
    per_class_iou = iu
    return float(miou), float(pixacc), per_class_iou.tolist()

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

# ---------------- Torch helpers ----------------

def upsample_nearest(t, size_hw):
    # t: (B,1,H',W') or (B,H',W') -> upsample to (B, H, W)
    if t.dim() == 3: t = t.unsqueeze(1)
    return F.interpolate(t.float(), size=size_hw, mode="nearest").squeeze(1).to(torch.long)
