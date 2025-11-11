# Networks/Architectures/unet.py
import torch
import torch.nn as nn

def conv_block(in_ch: int, out_ch: int):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    """
    Encoder-Decoder with skip connections.
    For supervised segmentation (Part A): out_classes = 5
    For SSL auto-encoder (Part B): out_classes = 3 (RGB reconstruction)
    """
    def __init__(self, in_ch: int = 3, out_classes: int = 5, feats=(64, 128, 256, 512)):
        super().__init__()
        f1, f2, f3, f4 = feats

        # Encoder
        self.down1 = conv_block(in_ch, f1)         # 64x64
        self.pool1 = nn.MaxPool2d(2)               # 32x32
        self.down2 = conv_block(f1, f2)
        self.pool2 = nn.MaxPool2d(2)               # 16x16
        self.down3 = conv_block(f2, f3)
        self.pool3 = nn.MaxPool2d(2)               # 8x8
        self.bott  = conv_block(f3, f4)

        # Decoder
        self.up3   = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)  # 8->16
        self.dec3  = conv_block(f4, f3)
        self.up2   = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)  # 16->32
        self.dec2  = conv_block(f3, f2)
        self.up1   = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)  # 32->64
        self.dec1  = conv_block(f2, f1)

        # Head
        self.head  = nn.Conv2d(f1, out_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        # Encoder
        x1 = self.down1(x)              # (B,f1,64,64)
        x2 = self.down2(self.pool1(x1)) # (B,f2,32,32)
        x3 = self.down3(self.pool2(x2)) # (B,f3,16,16)
        xb = self.bott(self.pool3(x3))  # (B,f4, 8, 8)  <-- features denses

        # Decoder
        y  = self.up3(xb)
        y  = self.dec3(torch.cat([y, x3], dim=1))
        y  = self.up2(y)
        y  = self.dec2(torch.cat([y, x2], dim=1))
        y  = self.up1(y)
        y  = self.dec1(torch.cat([y, x1], dim=1))

        out = self.head(y)

        if return_feats:
            return out, xb
        return out
