import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop)
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch, p_drop=p_drop)
    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetLight(nn.Module):
    def __init__(self, in_ch=3, n_classes=5, base_ch=32, dropout=0.1):
        super().__init__()
        c = base_ch
        self.enc1 = ConvBlock(in_ch, c, p_drop=dropout); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c, c*2, p_drop=dropout);   self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(c*2, c*4, p_drop=dropout); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c*4, c*8, p_drop=dropout)
        self.up3 = UpBlock(c*8, c*4, p_drop=dropout)
        self.up2 = UpBlock(c*4, c*2, p_drop=dropout)
        self.up1 = UpBlock(c*2, c, p_drop=dropout)
        self.head = nn.Conv2d(c, n_classes, kernel_size=1)
    def forward(self, x):
        s1 = self.enc1(x); x = self.pool1(s1)
        s2 = self.enc2(x); x = self.pool2(s2)
        s3 = self.enc3(x); x = self.pool3(s3)
        x = self.bottleneck(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)
