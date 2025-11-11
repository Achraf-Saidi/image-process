# Networks/Architectures/basicNetwork.py
import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
    """
    Compatible avec:
      - BasicNet(out_classes=5, in_ch=3, hidden=64)
      - BasicNet(param=cfg)  # où cfg["MODEL"]["NB_CHANNEL"] existe
    """
    def __init__(self, in_ch: int = 3, out_classes: int = 5, hidden: int = None, param: dict = None):
        super().__init__()

        # Récupère hidden depuis param si fourni
        if param is not None and hidden is None:
            hidden = int(param.get("MODEL", {}).get("NB_CHANNEL", 64))
        if hidden is None:
            hidden = 64

        # padding=2 <=> 'same' pour kernel_size=5 (plus robuste selon version PyTorch)
        self.conv1 = nn.Conv2d(in_ch,  hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hidden, out_classes, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return self.conv3(x)
