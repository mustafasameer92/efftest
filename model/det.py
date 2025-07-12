import torch
import torch.nn as nn
from model.backbone import EfficientNet
from model.bifpn import BiFPN
from model.head import ClassificationHead, BoxRegressionHead

class EfficientDet(nn.Module):
    def __init__(self, model_name, num_classes=80):
        super().__init__()
        self.backbone = EfficientNet(model_name)
        self.conv_reduce = nn.Conv2d(40, 64, 1)
        self.bifpn = BiFPN(64)
        self.class_head = ClassificationHead(64, num_anchors=9, num_classes=num_classes)
        self.box_head = BoxRegressionHead(64, num_anchors=9)

    def forward(self, x):
        feats = self.backbone(x)
        feats = [self.conv_reduce(f) for f in feats]
        feats += [nn.MaxPool2d(2)(feats[-1]), nn.MaxPool2d(4)(feats[-1])]
        feats = self.bifpn(feats[:3])  # P3, P4, P5
        class_out = self.class_head(feats)
        box_out = self.box_head(feats)
        return class_out, box_out

    @staticmethod
    def from_name(name):
        return EfficientDet(model_name=name)

    @staticmethod
    def from_pretrained(name):
        model = EfficientDet(model_name=name)
        # we can upload weight
        return model
