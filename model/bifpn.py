import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import DepthWiseSeparableConvModule as DWSConv
from model.module import MaxPool2dSamePad

class BiFPN(nn.Module):
    def __init__(self, channels):
        super(BiFPN, self).__init__()
        self.eps = 1e-4
        self.w3 = nn.Parameter(torch.ones(2))
        self.w4 = nn.Parameter(torch.ones(2))
        self.w5 = nn.Parameter(torch.ones(2))

        self.conv3 = DWSConv(channels, channels, relu=False)
        self.conv4 = DWSConv(channels, channels, relu=False)
        self.conv5 = DWSConv(channels, channels, relu=False)

        self.down = MaxPool2dSamePad(3, 2)
        self.up = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='nearest')

    def _fuse(self, weights, features):
        w = F.softplus(weights)
        return sum(w[i] * f for i, f in enumerate(features)) / (w.sum() + self.eps)

    def forward(self, features):
        p3, p4, p5 = features
        p4_td = self._fuse(self.w4, [p4, self.up(p5, p4)])
        p3_out = self.conv3(self._fuse(self.w3, [p3, self.up(p4_td, p3)]))
        p4_out = self.conv4(self._fuse(self.w4, [p4, p4_td, self.down(p3_out)]))
        p5_out = self.conv5(self._fuse(self.w5, [p5, self.down(p4_out)]))
        return [p3_out, p4_out, p5_out]
