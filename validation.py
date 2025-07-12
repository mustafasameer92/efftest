import torch
from config import *
from dataloader import get_loader

def validate(model, device, args):
    model.eval()
    val_loader = get_loader(VAL_SET, VAL_ANNOTATIONS, BATCH_SIZE)
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            class_out, box_out = model(imgs)
            print("Validation batch complete.")
