# EfficientDet-D0 (Custom Implementation for COCO 2017)

This project is a clean and enhanced implementation of **EfficientDet-D0**, modified for performance and structure clarity. It is optimized for training on the **COCO 2017 dataset**, using:

- A simplified EfficientNet-B0 backbone (block2a only)
- A 2-layer BiFPN with Softplus-based feature fusion
- Custom heads for box regression and classification
- PyTorch-based training pipeline with DataLoader and COCO annotation support
- FP16-ready design (no quantization)

---

## 🗂️ Project Structure

```
efficientdet_d0/
├── main.py
├── train.py
├── validation.py
├── dataloader.py
├── config.py
├── utils/
│   ├── utils.py
│   ├── tools.py
│   └── transforms.py
├── model/
│   ├── __init__.py
│   ├── det.py
│   ├── backbone.py
│   ├── bifpn.py
│   ├── head.py
│   ├── module.py
│   └── efficientnet/
│       ├── __init__.py
│       └── utils.py
├── dataset/
│   └── coco/
│       ├── images/
│       │   ├── train2017/
│       │   └── val2017/
│       └── annotations/
│           ├── instances_train2017.json
│           └── instances_val2017.json
```

---

## 🚀 How to Train

Make sure your environment includes:

- PyTorch
- torchvision
- pycocotools
- Python 3.8+

Then run:

```bash
python3 main.py -mode trainval -model efficientdet-d0 --cuda
```

---

## 🔧 Configuration

You can modify training paths and parameters in `config.py`:

```python
TRAIN_SET = dataset/coco/images/train2017
VAL_SET = dataset/coco/images/val2017
TRAIN_ANNOTATIONS = dataset/coco/annotations/instances_train2017.json
VAL_ANNOTATIONS = dataset/coco/annotations/instances_val2017.json
BATCH_SIZE = 4
NUM_CLASSES = 80
```

---

## ✅ Features

- Modular code structure
- BiFPN with learnable weights (Softplus fusion)
- EfficientNet-B0 as base backbone
- Compatible with COCO evaluation tools

---

## 📌 Notes

- Loss functions and evaluation metrics should be plugged in as needed (currently placeholder).
- Anchors generation and NMS are not implemented in this minimal version (add if needed).

---

## ✨ Author

mustafa ahmed
