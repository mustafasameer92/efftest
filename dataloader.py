from pycocotools.coco import COCO
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import cv2
from PIL import Image

class COCODataset:
    def __init__(self, path, annotation):
        self.coco = COCO(annotation)
        self.image_ids = list(self.coco.imgs.keys())
        self.path = path
        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.path, path)).convert("RGB")
        return self.transform(img), anns

    def __len__(self):
        return len(self.image_ids)

def get_loader(path, annotation, batch_size):
    dataset = COCODataset(path, annotation)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
