import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import MemoryEfficientSwish

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self._bn0 = nn.BatchNorm2d(32)
        self._swish = MemoryEfficientSwish()
        self._blocks = nn.ModuleList([
            nn.Conv2d(32, 40, kernel_size=3, padding=1),
            nn.Conv2d(40, 40, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        x = self._swish(self._bn0(self._conv_stem(x)))
        features = []
        for idx, block in enumerate(self._blocks):
            x = block(x)
            if idx == 1:
                features.append(x)
        return features

    @staticmethod
    def from_pretrained(model_name):
        return EfficientNet()
