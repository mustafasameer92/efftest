import torch
import torch.nn as nn
from model.efficientnet import EfficientNet as EffNet

class EfficientNet(nn.Module):
    def __init__(self, model_name):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(model_name)
        del model._fc, model._avg_pooling, model._dropout
        self.model = model

    def forward(self, x):
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        for idx, block in enumerate(self.model._blocks):
            x = block(x)
            if idx == 2:
                return [x]  # block2a: 40 channels
