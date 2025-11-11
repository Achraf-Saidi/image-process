# Networks/Architectures/unet_plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch, s=1, d=1):
    p = d
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=s, padding=p, dilation=d, bias=False)

class SCSE(nn.Module):
    """Squeeze-and-Excitation (channel) + spatial excitation."""
    def __init__(self, ch, r=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch//r, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch//r, 1), ch, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, d=1, drop=0.1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, d=d)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, d=d)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch!=out_ch else nn.Identity()
        self.scse  = SCSE(out_ch)
        self.drop  = nn.Dropout2d(drop) if drop>0 else nn.Identity()
    def forward(self, x):
        r = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.drop(F.relu(self.bn2(self.conv2(x)), inplace=True))
        x = self.scse(x) + r
        return F.relu(x, inplace=True)

class ASPP(nn.Module):
    def __init__(self, ch, out_ch):
        super().__init__()
        self.b1 = nn.Conv2d(ch, out_ch, 1, bias=False)
        self.b2 = conv3x3(ch, out_ch, d=2)
        self.b3 = conv3x3(ch, out_ch, d=4)
        self.b4 = conv3x3(ch, out_ch, d=6)
        self.proj = nn.Conv2d(out_ch*4, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        xs = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        x = torch.cat(xs, dim=1)
        return F.relu(self.bn(self.proj(x)), inplace=True)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
    def forward(self, x): return self.up(x)

class UNetPlus(nn.Module):
    """
    U-Net résiduel + SCSE + ASPP au bottleneck.
    in_ch = 3 (RGB) ou 4 si ExG est activé.
    """
    def __init__(self, in_ch=3, out_classes=5, feats=(64,128,256,512), drop=0.1):
        super().__init__()
        f1,f2,f3,f4 = feats
        # Encoder
        self.e1 = ResBlock(in_ch, f1, drop=drop)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ResBlock(f1, f2, drop=drop)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = ResBlock(f2, f3, drop=drop)
        self.p3 = nn.MaxPool2d(2)
        self.bott = ASPP(f3, f4)

        # Decoder
        self.up3 = Up(f4, f3); self.d3 = ResBlock(f3+f3, f3, drop=drop)
        self.up2 = Up(f3, f2); self.d2 = ResBlock(f2+f2, f2, drop=drop)
        self.up1 = Up(f2, f1); self.d1 = ResBlock(f1+f1, f1, drop=drop)

        self.head = nn.Conv2d(f1, out_classes, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.p1(x1))
        x3 = self.e3(self.p2(x2))
        xb = self.bott(self.p3(x3))
        y  = self.d3(torch.cat([self.up3(xb), x3], dim=1))
        y  = self.d2(torch.cat([self.up2(y),  x2], dim=1))
        y  = self.d1(torch.cat([self.up1(y),  x1], dim=1))
        return self.head(y)
