import torch.nn as nn
import torch.nn.functional as F

class DepthWiseSeparableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class MaxPool2dSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding=kernel_size // 2)

    def forward(self, x):
        return self.pool(x)
