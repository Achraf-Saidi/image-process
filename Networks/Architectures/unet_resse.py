# Networks/Architectures/unet_resse.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        h = max(ch // r, 4)
        self.fc1 = nn.Conv2d(ch, h, 1)
        self.fc2 = nn.Conv2d(h, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ResBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch)
        self.drop  = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.skip  = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = self.drop(x)
        x = F.relu(x + identity, inplace=True)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = ResBlockSE(in_ch, out_ch, p_drop=p_drop)  # in_ch = out_ch + skip_ch

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class UNetResSE(nn.Module):
    """
    task:
      - "seg"   => logits out_ch classes (no activation)
      - "recon" => reconstruct RGB in [0,1] (sigmoid)
    """
    def __init__(self, in_ch=3, out_ch=5, base_ch=48, dropout=0.1, task="seg", proj_dim=64):
        super().__init__()
        self.task = task

        c = base_ch
        self.enc1 = ResBlockSE(in_ch, c,   p_drop=dropout); self.pool1 = nn.MaxPool2d(2)   # 64 -> 32
        self.enc2 = ResBlockSE(c,   c*2, p_drop=dropout); self.pool2 = nn.MaxPool2d(2)   # 32 -> 16
        self.enc3 = ResBlockSE(c*2, c*4, p_drop=dropout); self.pool3 = nn.MaxPool2d(2)   # 16 -> 8
        self.bottleneck = ResBlockSE(c*4, c*8, p_drop=dropout)                               # 8

        self.up3 = UpBlock(c*8, c*4, p_drop=dropout)  # 8 -> 16
        self.up2 = UpBlock(c*4, c*2, p_drop=dropout)  # 16 -> 32
        self.up1 = UpBlock(c*2, c,   p_drop=dropout)  # 32 -> 64

        self.head = nn.Conv2d(c, out_ch, 1)

        # projection head for SSL clustering (reduces channel dim)
        self.proj = nn.Conv2d(c*2, proj_dim, 1)  # default: use enc2 feature map (C=2*base)

    def forward(self, x):
        s1 = self.enc1(x); x = self.pool1(s1)
        s2 = self.enc2(x); x = self.pool2(s2)
        s3 = self.enc3(x); x = self.pool3(s3)
        x = self.bottleneck(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        y = self.head(x)
        if self.task == "recon":
            return torch.sigmoid(y)
        return y

    @torch.no_grad()
    def encode(self, x, level="enc2", project=True):
        """
        Returns a feature map (C,H',W') suitable for clustering.
        level: enc1 (32x32), enc2 (16x16), enc3 (8x8), bottleneck (8x8)
        """
        s1 = self.enc1(x); x1 = self.pool1(s1)
        s2 = self.enc2(x1); x2 = self.pool2(s2)
        s3 = self.enc3(x2); x3 = self.pool3(s3)
        b  = self.bottleneck(x3)

        feat = {"enc1": s1, "enc2": s2, "enc3": s3, "bottleneck": b}.get(level, s2)
        if project and level == "enc2":
            feat = self.proj(feat)
        return feat
