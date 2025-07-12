import torch
import torch.nn as nn
from model.module import DepthWiseSeparableConvModule as DWSConv

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            DWSConv(in_channels, in_channels),
            DWSConv(in_channels, in_channels)
        )
        self.output = nn.Conv2d(in_channels, num_anchors * num_classes, 1)

    def forward(self, features):
        outputs = [self.output(self.conv(f)) for f in features]
        return torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1, -1) for o in outputs], dim=1)

class BoxRegressionHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Sequential(
            DWSConv(in_channels, in_channels),
            DWSConv(in_channels, in_channels)
        )
        self.output = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(self, features):
        outputs = [self.output(self.conv(f)) for f in features]
        return torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1, 4) for o in outputs], dim=1)
