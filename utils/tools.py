import torch

def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
