import os, glob, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import albumentations.pytorch as AT
from utils import parse_rr_cc, discover_grid, is_test_tile

CLASSES = 5
TILE_H = 64
TILE_W = 64

def collect_paths(root_images, root_masks):
    img_paths = sorted(glob.glob(os.path.join(root_images, "tile_*.png")))
    msk_paths = sorted(glob.glob(os.path.join(root_masks, "tile_*.png")))
    assert len(img_paths)==len(msk_paths)>0, "Images/masks mismatch or empty."
    for a,b in zip(img_paths, msk_paths):
        assert os.path.basename(a)==os.path.basename(b), f"Pairing error: {a} vs {b}"
    return img_paths, msk_paths

def split_sets(img_paths, seed=42, train_ratio=0.8):
    n_rows, n_cols = discover_grid([os.path.basename(p) for p in img_paths])
    test_idx, rest = [], []
    for i, p in enumerate(img_paths):
        rr, cc = parse_rr_cc(os.path.basename(p))
        (test_idx if is_test_tile(rr, cc, n_rows, n_cols) else rest).append(i)
    rng = random.Random(seed); rng.shuffle(rest)
    n_train = int(len(rest)*train_ratio)
    return rest[:n_train], rest[n_train:], test_idx, n_rows, n_cols

def build_tfms(split, aug_cfg):
    if split=="train" and aug_cfg.get("USE_AUG", True):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=0),
            A.ColorJitter(0.2,0.2,0.2,0.02, p=0.5),
            A.GaussNoise(var_limit=(5,20), p=0.2),
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            AT.ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            AT.ToTensorV2()
        ])

class LLNDataset(Dataset):
    def __init__(self, root="Dataset", split="train", indices=None, aug_cfg=None, add_exg=False):
        self.img_paths, self.msk_paths = collect_paths(os.path.join(root,"images"), os.path.join(root,"annotations"))
        self.split = split
        self.indices = indices if indices is not None else list(range(len(self.img_paths)))
        self.tfms = build_tfms(split, aug_cfg or {})
        self.add_exg = add_exg
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        j = self.indices[idx]
        img_p = self.img_paths[j]; msk_p = self.msk_paths[j]
        img = np.array(Image.open(img_p).convert("RGB"))
        msk = np.array(Image.open(msk_p))
        out = self.tfms(image=img, mask=msk)
        x = out["image"]               # (3,64,64) float[-1,1]
        m = out["mask"]                # Tensor long / uint8
        y = m.long() if torch.is_tensor(m) else torch.from_numpy(m.astype(np.int64))
        if self.add_exg:
            # ExG sur image originale (pas normalisée) -> remet en [-1,1]
            R,G,B = img[...,0].astype(np.float32), img[...,1].astype(np.float32), img[...,2].astype(np.float32)
            exg = (2*G - R - B)
            exg = (exg - exg.min()) / (exg.max()-exg.min() + 1e-6)
            exg = (exg - 0.5)/0.5
            exg_t = torch.from_numpy(exg).float().unsqueeze(0)
            x = torch.cat([x, exg_t], dim=0)     # (4,64,64)
        name = os.path.basename(img_p)
        return x, y, name

def make_loaders(cfg):
    seed = cfg.get("SEED", 42)
    tr_ratio = cfg.get("TRAIN_RATIO", 0.8)
    bs = cfg.get("BATCH_SIZE", 128)
    nw = cfg.get("NUM_WORKERS", 0)
    ds_root = cfg.get("DATA_ROOT", "Dataset")
    use_exg = cfg.get("ADD_EXG", False)
    and if the cons
    

    all_imgs,_ = collect_paths(os.path.join(ds_root,"images"), os.path.join(ds_root,"annotations"))
    train_idx, val_idx, test_idx, n_rows, n_cols = split_sets(all_imgs, seed=seed, train_ratio=tr_ratio)

    aug = cfg.get("AUG", {"USE_AUG": True})
    dtr = LLNDataset(ds_root,"train",train_idx,aug,add_exg=use_exg)
    dva = LLNDataset(ds_root,"val",  val_idx, {"USE_AUG": False}, add_exg=use_exg)
    dte = LLNDataset(ds_root,"test", test_idx, {"USE_AUG": False}, add_exg=use_exg)

    Ltr = DataLoader(dtr, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True, )
    Lva = DataLoader(dva, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    Lte = DataLoader(dte, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return Ltr, Lva, Lte, (n_rows, n_cols)
    

