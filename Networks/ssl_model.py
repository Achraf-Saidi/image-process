# Networks/ssl_model.py — Part B (SSL autoencoder + clustering + mosaïques)

import os, json, math, glob, re, time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.cuda import amp

try:
    from Networks.Architectures.ae_unet_light import AEUNetLight
except Exception:
    from .Architectures.ae_unet_light import AEUNetLight


CLASS_NAMES = ["others", "water", "buildings", "farmlands", "green spaces"]

# ============== utils ==============

def _ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def _get(cfg, path, default=None):
    cur = cfg
    for k in path.split('.'):
        if not isinstance(cur, dict): return default
        cur = cur.get(k, None)
    return default if cur is None else cur

def _plot_curve(vals_tr, vals_va, ylabel, out_path):
    e = range(1, len(vals_tr)+1)
    plt.figure()
    plt.plot(e, vals_tr, label="train")
    plt.plot(e, vals_va, label="val")
    plt.xlabel("epoch"); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def _mask_to_color(mask):
    palette = np.array([
        [180, 180, 180],  # others
        [  0,  90, 200],  # water
        [200,   0,   0],  # buildings
        [230, 190,  30],  # farmlands
        [ 40, 160,  40],  # green spaces
    ], dtype=np.uint8)
    h, w = mask.shape
    return palette[mask.reshape(-1)].reshape(h, w, 3)

def _overlay(rgb_uint8, mask_rgb_uint8, alpha=0.45):
    rgb = rgb_uint8.astype(np.float32)
    m = mask_rgb_uint8.astype(np.float32)
    out = (1-alpha)*rgb + alpha*m
    return np.clip(out, 0, 255).astype(np.uint8)

_RC_RE   = re.compile(r".*r(\d+)[_-]c(\d+).*", re.IGNORECASE)
_TILE_RE = re.compile(r".*tile[_-]?(\d+)[_-](\d+).*", re.IGNORECASE)

def _parse_rc(filename):
    b = os.path.basename(filename)
    m = _RC_RE.match(b)
    if m: return int(m.group(1)), int(m.group(2))
    m = _TILE_RE.match(b)
    if m: return int(m.group(1)), int(m.group(2))
    return None, None

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

def _metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / (tp + fp + fn + 1e-9)
    dice = 2*tp / (2*tp + fp + fn + 1e-9)
    acc = tp.sum() / (cm.sum() + 1e-9)
    return {
        "pixel_acc": float(acc),
        "iou_per_class": iou.tolist(),
        "miou": float(np.mean(iou)),
        "dice_per_class": dice.tolist(),
        "mdice": float(np.mean(dice))
    }

def _best_perm_mapping(cm, K):
    """
    Unsupervised eval trick: choose cluster->class mapping that maximizes mean IoU.
    K=5 => brute-force 5! =120 (cheap, no scipy needed).
    cm: confusion matrix with rows=GT classes, cols=clusters
    """
    import itertools
    best = None
    best_score = -1
    for perm in itertools.permutations(range(K)):
        # map cluster j -> class perm[j]  => need cm_mapped over classes
        # Equivalent: reorder columns by inverse mapping:
        # predicted mapped class c comes from cluster j where perm[j]==c
        inv = [perm.index(c) for c in range(K)]
        cm2 = cm[:, inv]  # columns become mapped classes
        mets = _metrics_from_cm(cm2)
        if mets["miou"] > best_score:
            best_score = mets["miou"]
            best = perm
    # best is cluster->class mapping
    return list(best), best_score

# ============== datasets ==============

class SSLImageOnlyDataset(Dataset):
    def __init__(self, pairs, resize_hw=None, augment=False):
        self.pairs = pairs
        self.resize_hw = resize_hw
        self.augment = augment

    def __len__(self): return len(self.pairs)

    def _read_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.resize_hw is not None:
            img = img.resize((self.resize_hw[1], self.resize_hw[0]), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)/255.0
        x = torch.from_numpy(arr).permute(2,0,1)  # CHW
        return x

    def _aug(self, x):
        # simple + safe for reconstruction
        if np.random.rand() < 0.5:
            x = torch.flip(x, dims=[2])  # Hflip
        k = np.random.randint(0,4)
        if k:
            x = torch.rot90(x, k, dims=[1,2])
        # mild brightness/contrast jitter
        if np.random.rand() < 0.7:
            mean = x.mean(dim=(1,2), keepdim=True)
            c = float(np.random.uniform(0.9, 1.1))
            b = float(np.random.uniform(0.9, 1.1))
            x = (x - mean)*c + mean
            x = (x*b).clamp(0,1)
        return x

    def __getitem__(self, idx):
        ip, _ = self.pairs[idx]
        x = self._read_image(ip)
        if self.augment:
            x = self._aug(x)
        return x

class SegmEvalDataset(Dataset):
    """Pour visualiser+évaluer (images + masks)."""
    def __init__(self, pairs, resize_hw=None):
        self.pairs = pairs
        self.resize_hw = resize_hw

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")
        if self.resize_hw is not None:
            img = img.resize((self.resize_hw[1], self.resize_hw[0]), Image.BILINEAR)
            msk = msk.resize((self.resize_hw[1], self.resize_hw[0]), Image.NEAREST)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)/255.0).permute(2,0,1)
        y = torch.from_numpy(np.asarray(msk, dtype=np.int64))
        return x, y, os.path.splitext(os.path.basename(ip))[0]

# ============== KMeans (sklearn optional, torch fallback) ==============

def _fit_kmeans(samples, K, seed=42):
    """
    samples: (N,D) float32 numpy
    returns centers: (K,D) float32 numpy
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=4096, n_init="auto")
        km.fit(samples)
        return km.cluster_centers_.astype(np.float32)
    except Exception:
        # tiny minibatch kmeans fallback
        rng = np.random.RandomState(seed)
        N, D = samples.shape
        centers = samples[rng.choice(N, K, replace=False)].copy()
        counts = np.zeros(K, dtype=np.float32)

        for it in range(200):
            batch = samples[rng.choice(N, min(4096, N), replace=False)]
            # distances
            # (B,K) = ||x||^2 -2 x c + ||c||^2
            x2 = (batch**2).sum(1, keepdims=True)
            c2 = (centers**2).sum(1, keepdims=True).T
            dist = x2 - 2*batch@centers.T + c2
            lab = dist.argmin(1)
            for k in range(K):
                sel = batch[lab==k]
                if sel.shape[0] == 0:
                    continue
                counts[k] += sel.shape[0]
                eta = sel.shape[0] / max(counts[k], 1.0)
                centers[k] = (1-eta)*centers[k] + eta*sel.mean(0)
        return centers.astype(np.float32)

@torch.no_grad()
def _assign_clusters_torch(feat_bchw, centers_kd):
    """
    feat_bchw: torch (B,C,H,W) float
    centers_kd: torch (K,C) float
    returns (B,H,W) long
    """
    B,C,H,W = feat_bchw.shape
    x = feat_bchw.permute(0,2,3,1).reshape(-1, C)  # (BHW,C)
    # distances via ||x||^2 - 2 x c + ||c||^2
    x2 = (x*x).sum(1, keepdim=True)              # (BHW,1)
    c2 = (centers_kd*centers_kd).sum(1).view(1,-1)  # (1,K)
    dist = x2 - 2.0*(x @ centers_kd.t()) + c2    # (BHW,K)
    lab = dist.argmin(1).view(B,H,W)
    return lab

# ============== Network Class used by main_ssl.py ==============

class Network_Class:
    def __init__(self, param, img_dir, mask_dir, results_path):
        self.cfg = param or {}
        self.results_dir = results_path
        _ensure_dir(self.results_dir)
        _ensure_dir(os.path.join(self.results_dir, "ssl"))
        _ensure_dir(os.path.join(self.results_dir, "mosaic"))

        # config
        self.num_classes = int(_get(self.cfg, "DATA.NUM_CLASSES", 5))
        resize = str(_get(self.cfg, "DATASET.RESIZE_SHAPE", "64x64")).lower()
        h,w = resize.split("x")
        self.tile_h, self.tile_w = int(h), int(w)
        self.resize_hw = (self.tile_h, self.tile_w)

        dev = str(_get(self.cfg, "DEVICE", "cpu")).lower()
        self.device = torch.device("cuda" if (dev=="cuda" and torch.cuda.is_available()) else "cpu")
        self.use_amp = bool(_get(self.cfg, "SSL.USE_AMP", True)) and (self.device.type=="cuda")
        self.scaler = amp.GradScaler(enabled=self.use_amp)

        # data pairing + split (quadrant lower-right = test)
        pairs = _pair_images_masks(img_dir, mask_dir)
        if not pairs:
            raise RuntimeError("No (image, mask) pairs found. Check Dataset/images and Dataset/annotations structure.")

        coords = [(_parse_rc(ip)[0], _parse_rc(ip)[1]) for ip,_ in pairs]
        if any(r is None for r,_ in coords) or any(c is None for _,c in coords):
            # fallback random (should not happen if tile_RR_CC naming is correct)
            rng = np.random.RandomState(42)
            idx = np.arange(len(pairs)); rng.shuffle(idx)
            n_test = int(0.25*len(pairs))
            test_idx = idx[:n_test]
            rest = idx[n_test:]
            n_val = int(0.20*len(rest))
            val_idx = rest[:n_val]; train_idx = rest[n_val:]
            split_mode = "FALLBACK random"
        else:
            rows = np.array([r for r,_ in coords]); cols = np.array([c for _,c in coords])
            r_thr = rows.max()*0.5; c_thr = cols.max()*0.5
            test_idx = np.where((rows >= r_thr) & (cols >= c_thr))[0]
            rest_idx = np.setdiff1d(np.arange(len(pairs)), test_idx)
            rng = np.random.RandomState(42); rng.shuffle(rest_idx)
            n_val = int(0.20*len(rest_idx))
            val_idx = rest_idx[:n_val]; train_idx = rest_idx[n_val:]
            split_mode = "QUADRANT (test=lower-right)"

        self.train_pairs = [pairs[i] for i in train_idx]
        self.val_pairs   = [pairs[i] for i in val_idx]
        self.test_pairs  = [pairs[i] for i in test_idx]
        self.all_pairs   = pairs

        print(f"[Split] mode={split_mode} | train={len(self.train_pairs)} val={len(self.val_pairs)} test={len(self.test_pairs)}")
        print(f"[Device] {self.device} | AMP={self.use_amp}")

        # loaders
        bs = int(_get(self.cfg, "SSL.BATCH_SIZE", 64))
        nw = int(_get(self.cfg, "SSL.NUM_WORKERS", 4))
        pin = (self.device.type=="cuda")

        self.ssl_train_loader = DataLoader(
            SSLImageOnlyDataset(self.train_pairs, resize_hw=self.resize_hw, augment=True),
            batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin
        )
        self.ssl_val_loader = DataLoader(
            SSLImageOnlyDataset(self.val_pairs, resize_hw=self.resize_hw, augment=False),
            batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin
        )

        self.eval_val_loader = DataLoader(
            SegmEvalDataset(self.val_pairs, resize_hw=self.resize_hw),
            batch_size=8, shuffle=False, num_workers=0
        )
        self.eval_test_loader = DataLoader(
            SegmEvalDataset(self.test_pairs, resize_hw=self.resize_hw),
            batch_size=8, shuffle=False, num_workers=0
        )

        # model
        base_ch = int(_get(self.cfg, "SSL.BASE_CH", 48))
        drop = float(_get(self.cfg, "SSL.DROPOUT", 0.1))
        self.net = AEUNetLight(in_ch=3, base_ch=base_ch, dropout=drop).to(self.device)

        # loss + opt
        lr = float(_get(self.cfg, "SSL.LR", 1e-3))
        wd = float(_get(self.cfg, "SSL.WEIGHT_DECAY", 1e-5))
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)
        self.epochs = int(_get(self.cfg, "SSL.EPOCHS", 60))
        self.patience = int(_get(self.cfg, "SSL.PATIENCE", 10))

        self.lambda_l1 = float(_get(self.cfg, "SSL.LAMBDA_L1", 0.5))

        self.best_state = None

    def _recon_loss(self, recon, x):
        # force float32 for stability
        recon = recon.float()
        x = x.float()
        mse = torch.mean((recon - x)**2)
        l1  = torch.mean(torch.abs(recon - x))
        return mse + self.lambda_l1*l1

    def train(self):
        tr_losses, va_losses = [], []
        best = math.inf
        no_imp = 0

        for ep in range(1, self.epochs+1):
            t0 = time.time()
            # --- train ---
            self.net.train()
            tot = 0.0
            n = 0
            for x in self.ssl_train_loader:
                x = x.to(self.device, non_blocking=True)
                with amp.autocast(enabled=self.use_amp):
                    recon = self.net(x)
                    loss = self._recon_loss(recon, x)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                tot += float(loss.item()) * x.size(0)
                n += x.size(0)

            tr = tot/max(n,1)

            # --- val ---
            self.net.eval()
            tot = 0.0
            n = 0
            with torch.no_grad():
                for x in self.ssl_val_loader:
                    x = x.to(self.device, non_blocking=True)
                    with amp.autocast(enabled=self.use_amp):
                        recon = self.net(x)
                        loss = self._recon_loss(recon, x)
                    tot += float(loss.item()) * x.size(0)
                    n += x.size(0)

            va = tot/max(n,1)
            tr_losses.append(tr); va_losses.append(va)

            _plot_curve(tr_losses, va_losses, "reconstruction loss",
                        os.path.join(self.results_dir, "ssl", "curves_recon_loss.png"))

            improved = (va < best - 1e-9)
            if improved:
                best = va
                self.best_state = {k: v.detach().cpu() for k,v in self.net.state_dict().items()}
                torch.save(self.best_state, os.path.join(self.results_dir, "ssl", "best_ae.pt"))
                no_imp = 0
            else:
                no_imp += 1

            self._save_recon_samples(ep, max_samples=6)

            print(f"[SSL {ep}/{self.epochs}] train_loss={tr:.4f} | val_loss={va:.4f} "
                  f"| best={best:.4f} {'<<' if improved else ''} | no_improve={no_imp} | {time.time()-t0:.1f}s")

            if no_imp >= self.patience:
                print(f"Early stop SSL (no improvement for {self.patience} epochs).")
                break

        # reload best
        if self.best_state is not None:
            self.net.load_state_dict(self.best_state, strict=True)

        # after training: cluster + eval visuals
        self.evaluate()

    def loadWeights(self):
        p = os.path.join(self.results_dir, "ssl", "best_ae.pt")
        if os.path.exists(p):
            self.net.load_state_dict(torch.load(p, map_location=self.device), strict=True)

    def _save_recon_samples(self, epoch, max_samples=6):
        self.net.eval()
        out = []
        saved = 0
        with torch.no_grad():
            for x in self.ssl_val_loader:
                x = x.to(self.device)
                recon = self.net(x).detach().cpu().clamp(0,1)
                x = x.detach().cpu().clamp(0,1)
                for i in range(x.size(0)):
                    if saved >= max_samples: break
                    out.append(x[i]); out.append(recon[i])
                    saved += 1
                break

        if saved == 0: return
        # grid 2 columns (orig/recon)
        nrow = saved
        H,W = self.tile_h, self.tile_w
        canvas = np.ones((nrow*H, 2*W, 3), dtype=np.float32)
        for i in range(saved):
            a = out[2*i].permute(1,2,0).numpy()
            b = out[2*i+1].permute(1,2,0).numpy()
            canvas[i*H:(i+1)*H, 0:W] = a
            canvas[i*H:(i+1)*H, W:2*W] = b
        plt.figure(figsize=(6, 2*saved))
        plt.imshow(canvas); plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "ssl", f"recon_ep{epoch}.png"), dpi=160)
        plt.close()

    @torch.no_grad()
    def _sample_features_for_kmeans(self, loader, max_pixels=200000):
        """
        Returns numpy samples (N,C) for kmeans.
        Sampling random pixels from decoder features.
        """
        self.net.eval()
        samples = []
        total = 0
        for x, _, _ in loader:
            x = x.to(self.device)
            with amp.autocast(enabled=self.use_amp):
                _, feat = self.net(x, return_feat=True)
            feat = feat.float()  # (B,C,H,W)
            B,C,H,W = feat.shape
            feat_flat = feat.permute(0,2,3,1).reshape(-1, C)  # (BHW,C)
            # sample subset
            remaining = max_pixels - total
            if remaining <= 0:
                break
            take = min(remaining, feat_flat.shape[0]//4 + 1)  # sample ~25%
            idx = torch.randint(0, feat_flat.shape[0], (take,), device=feat_flat.device)
            samp = feat_flat[idx].detach().cpu().numpy()
            samples.append(samp)
            total += samp.shape[0]
            if total >= max_pixels:
                break
        if len(samples) == 0:
            raise RuntimeError("Could not sample features for kmeans.")
        return np.concatenate(samples, axis=0).astype(np.float32)

    @torch.no_grad()
    def _predict_clusters_loader(self, loader, centers_kc, mapping_cluster_to_class=None, save_quali_path=None):
        """
        Predict cluster masks on a loader with (x, y, name).
        Returns confusion matrix between GT classes and predicted clusters (or mapped classes if mapping provided).
        """
        K = centers_kc.shape[0]
        centers = torch.from_numpy(centers_kc).to(self.device).float()  # (K,C)

        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        saved = 0
        tiles = []

        for x, y, name in loader:
            x = x.to(self.device)
            y = y.numpy().astype(np.int64)  # (B,H,W)

            with amp.autocast(enabled=self.use_amp):
                _, feat = self.net(x, return_feat=True)
            feat = feat.float()

            clusters = _assign_clusters_torch(feat, centers).cpu().numpy()  # (B,H,W)

            if mapping_cluster_to_class is not None:
                # map clusters -> semantic class ids
                mapped = np.zeros_like(clusters)
                for cl in range(K):
                    mapped[clusters == cl] = mapping_cluster_to_class[cl]
                pred = mapped
            else:
                pred = clusters  # not comparable to GT without mapping

            # update confusion matrix (GT classes vs pred classes)
            for i in range(pred.shape[0]):
                gt = y[i].reshape(-1)
                pr = pred[i].reshape(-1)
                m = (gt>=0) & (gt<self.num_classes)
                idx = self.num_classes*gt[m] + pr[m]
                binc = np.bincount(idx, minlength=self.num_classes*self.num_classes)
                cm += binc.reshape(self.num_classes, self.num_classes)

            # qualitative few samples
            if save_quali_path is not None and saved < 12:
                x_cpu = (x.detach().cpu().clamp(0,1).permute(0,2,3,1).numpy()*255).astype(np.uint8)
                for i in range(pred.shape[0]):
                    if saved >= 12: break
                    rgb = x_cpu[i]
                    gt_rgb = _mask_to_color(y[i])
                    pr_rgb = _mask_to_color(pred[i])
                    tiles.append((rgb, gt_rgb, pr_rgb))
                    saved += 1

        if save_quali_path is not None and len(tiles)>0:
            # grid: RGB | GT | PRED
            n = len(tiles)
            H,W = self.tile_h, self.tile_w
            canvas = np.ones((n*H, 3*W, 3), dtype=np.uint8)*255
            for i,(rgb,gt,pr) in enumerate(tiles):
                canvas[i*H:(i+1)*H, 0:W] = rgb
                canvas[i*H:(i+1)*H, W:2*W] = gt
                canvas[i*H:(i+1)*H, 2*W:3*W] = pr
            plt.figure(figsize=(10, 2*n))
            plt.imshow(canvas); plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_quali_path, dpi=170)
            plt.close()

        return cm

    @torch.no_grad()
    def _save_mosaics(self, centers_kc, mapping):
        # needs r/c in filenames
        def collect(pairs):
            rc2 = {}
            rows, cols = [], []
            for ip, mp in pairs:
                r,c = _parse_rc(ip)
                if r is None or c is None: return None
                rc2[(r,c)] = (ip, mp)
                rows.append(r); cols.append(c)
            return rc2, (min(rows), max(rows), min(cols), max(cols))

        out_dir = os.path.join(self.results_dir, "mosaic")
        _ensure_dir(out_dir)

        allc = collect(self.all_pairs)
        if allc is None:
            print("[mosaic] filenames without tile row/col -> cannot assemble full mosaics.")
            return

        rc2, (rmin,rmax,cmin,cmax) = allc
        H = (rmax-rmin+1)*self.tile_h
        W = (cmax-cmin+1)*self.tile_w

        full_rgb = np.zeros((H,W,3), dtype=np.uint8)
        full_gt  = np.zeros((H,W), dtype=np.int64)

        for (r,c),(ip,mp) in rc2.items():
            y0 = (r-rmin)*self.tile_h
            x0 = (c-cmin)*self.tile_w
            img = Image.open(ip).convert("RGB").resize((self.tile_w,self.tile_h), Image.BILINEAR)
            msk = Image.open(mp).convert("L").resize((self.tile_w,self.tile_h), Image.NEAREST)
            full_rgb[y0:y0+self.tile_h, x0:x0+self.tile_w] = np.asarray(img, dtype=np.uint8)
            full_gt[y0:y0+self.tile_h, x0:x0+self.tile_w] = np.asarray(msk, dtype=np.int64)

        plt.figure(figsize=(10,10)); plt.imshow(full_rgb); plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "full_rgb.png"), dpi=200); plt.close()
        plt.figure(figsize=(10,10)); plt.imshow(_mask_to_color(full_gt)); plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "full_gt.png"), dpi=200); plt.close()

        # test quadrant mosaic
        testc = collect(self.test_pairs)
        if testc is None:
            print("[mosaic] cannot assemble test mosaic (no tile indices).")
            return

        rc2t, (rmin,rmax,cmin,cmax) = testc
        Ht = (rmax-rmin+1)*self.tile_h
        Wt = (cmax-cmin+1)*self.tile_w
        test_rgb  = np.zeros((Ht,Wt,3), dtype=np.uint8)
        test_gt   = np.zeros((Ht,Wt), dtype=np.int64)
        test_pred = np.zeros((Ht,Wt), dtype=np.int64)

        centers = torch.from_numpy(centers_kc).to(self.device).float()

        self.net.eval()
        for (r,c),(ip,mp) in rc2t.items():
            y0 = (r-rmin)*self.tile_h
            x0 = (c-cmin)*self.tile_w
            img = Image.open(ip).convert("RGB").resize((self.tile_w,self.tile_h), Image.BILINEAR)
            rgb_tile = np.asarray(img, dtype=np.uint8)
            test_rgb[y0:y0+self.tile_h, x0:x0+self.tile_w] = rgb_tile

            msk = Image.open(mp).convert("L").resize((self.tile_w,self.tile_h), Image.NEAREST)
            gt_tile = np.asarray(msk, dtype=np.int64)
            test_gt[y0:y0+self.tile_h, x0:x0+self.tile_w] = gt_tile

            x = torch.from_numpy(rgb_tile.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(self.device)
            _, feat = self.net(x, return_feat=True)
            feat = feat.float()
            cl = _assign_clusters_torch(feat, centers).cpu().numpy()[0]
            mapped = np.zeros_like(cl)
            for k in range(self.num_classes):
                mapped[cl==k] = mapping[k]
            test_pred[y0:y0+self.tile_h, x0:x0+self.tile_w] = mapped

        plt.figure(figsize=(10,10)); plt.imshow(_mask_to_color(test_gt)); plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "test_gt.png"), dpi=200); plt.close()
        plt.figure(figsize=(10,10)); plt.imshow(_mask_to_color(test_pred)); plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "test_pred.png"), dpi=200); plt.close()

        plt.figure(figsize=(10,10)); plt.imshow(_overlay(test_rgb, _mask_to_color(test_gt))); plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "test_overlay_gt.png"), dpi=200); plt.close()
        plt.figure(figsize=(10,10)); plt.imshow(_overlay(test_rgb, _mask_to_color(test_pred))); plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "test_overlay_pred.png"), dpi=200); plt.close()

    def evaluate(self):
        self.loadWeights()

        # 1) Fit KMeans on train features (no labels)
        max_pix = int(_get(self.cfg, "SSL.KMEANS_MAX_PIXELS", 200000))
        print(f"[KMeans] sampling up to {max_pix} pixels for clustering...")
        train_eval_loader = DataLoader(SegmEvalDataset(self.train_pairs, resize_hw=self.resize_hw),
                                       batch_size=8, shuffle=True, num_workers=0)
        samples = self._sample_features_for_kmeans(train_eval_loader, max_pixels=max_pix)
        centers = _fit_kmeans(samples, self.num_classes, seed=int(_get(self.cfg, "SSL.SEED", 42)))
        print("[KMeans] centers fitted:", centers.shape)

        # 2) Compute mapping on VAL (clusters -> classes) using brute-force best mIoU
        cm_val_clusters = self._predict_clusters_loader(
            self.eval_val_loader,
            centers,
            mapping_cluster_to_class=None,
            save_quali_path=None
        )
        # cm_val_clusters is GT vs clusters, need best permutation mapping clusters->classes
        mapping, score = _best_perm_mapping(cm_val_clusters, self.num_classes)
        # mapping is cluster->class
        print(f"[VAL mapping] best mIoU after mapping = {score:.3f}")
        with open(os.path.join(self.results_dir, "ssl", "cluster_mapping.json"), "w") as f:
            json.dump({"cluster_to_class": mapping, "val_miou_after_mapping": score}, f, indent=2)

        # 3) Evaluate on VAL with mapping (for report)
        cm_val = self._predict_clusters_loader(
            self.eval_val_loader,
            centers,
            mapping_cluster_to_class=mapping,
            save_quali_path=os.path.join(self.results_dir, "ssl", "quali_val_clusters.png")
        )
        mets_val = _metrics_from_cm(cm_val)

        # 4) Evaluate on TEST with mapping (main result)
        cm_test = self._predict_clusters_loader(
            self.eval_test_loader,
            centers,
            mapping_cluster_to_class=mapping,
            save_quali_path=os.path.join(self.results_dir, "ssl", "quali_test_clusters.png")
        )
        mets_test = _metrics_from_cm(cm_test)

        # save metrics
        with open(os.path.join(self.results_dir, "ssl", "metrics_val.json"), "w") as f:
            json.dump(mets_val, f, indent=2)
        with open(os.path.join(self.results_dir, "ssl", "metrics_test.json"), "w") as f:
            json.dump(mets_test, f, indent=2)

        # mosaics for presentation
        self._save_mosaics(centers, mapping)

        print(f"[SSL TEST] acc={mets_test['pixel_acc']:.3f} | mIoU={mets_test['miou']:.3f} | mDice={mets_test['mdice']:.3f}")
        return mets_test
