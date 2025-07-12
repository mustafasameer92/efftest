import torch
from config import *
from dataloader import get_loader

def train(model, device, args):
    model.train()
    train_loader = get_loader(TRAIN_SET, TRAIN_ANNOTATIONS, BATCH_SIZE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1):
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            class_out, box_out = model(imgs)
            loss = class_out.sum() * 0.0 + box_out.sum() * 0.0  # مؤقتًا بدون loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} finished.")
