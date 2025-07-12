# EfficientDet-D0 Enhanced (PyTorch, COCO2017)

This repository contains a fully customized and optimized implementation of **EfficientDet-D0**, enhanced for better accuracy and speed on the **COCO 2017** dataset.

---

## 🚀 Key Enhancements

- ✅ EfficientNet-B0 used as backbone, extracting only **block2a** output (40 channels)
- ✅ Channels reduced to 64 via `conv_reduce`
- ✅ **BiFPN** redesigned to use only **2 layers** with **Softplus-learnable weighted fusion**
- ✅ Only **P3, P4, P5** levels used (no P6/P7)
- ✅ Custom anchor generator for P3–P5
- ✅ FP16 training supported
- ✅ Optimized for **mAP@50** and real-time inference (low FLOPs, high FPS)

---

## 📁 Project Structure

```
.
├── backbone.py         # Custom EfficientNet-B0 (only block2a)
├── bifpn.py            # 2-layer BiFPN with Softplus fusion
├── head.py             # Classification and box regression heads
├── det.py              # Complete EfficientDet-D0 model
├── config.py           # Hyperparameters and model settings
├── train.py            # Training loop with FP16 + SGD + COCO
├── validation.py       # COCO mAP evaluation using pycocotools
├── dataloader.py       # COCO loader
├── transforms.py       # Data augmentation and normalization
├── processing.py       # Post-processing, decoding boxes
├── utils.py, tools.py  # Helper functions and loss
├── anchors.py          # Anchor generation for P3–P5
└── main.py             # Entry point for training / evaluation
```

---

## 🧠 Model Architecture

```text
Input Image 512x512
   ↓
EfficientNet-B0 (block2a)
   ↓
Conv1x1 Reduce (40→64 channels)
   ↓
P3 = output
P4 = MaxPool(P3)
P5 = MaxPool(P4)
   ↓
BiFPN (2 layers, Softplus fusion)
   ↓
Class / Box Heads (3 heads each)
```

---

## 📦 Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- torchvision
- pycocotools
- tqdm, opencv-python, Pillow, numpy

Install:
```bash
pip install -r requirements.txt
```

---

## 📂 COCO Dataset Setup

```
coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

---

## 🏋️‍♂️ Train the Model

```bash
python main.py -mode trainval -model efficientdet-d0 --cuda
```

---

## 🧪 Evaluate mAP@50

```bash
python main.py -mode eval -model efficientdet-d0 --cuda
```

---

## 📈 Expected Outcome

- +3% improvement in **mAP@50** over original EfficientDet-D0
- Reduced model **FLOPs**
- Faster inference with FP16 + Softplus
- COCO-compatible predictions using pycocotools

---

## 🔐 License

MIT — Free to use and modify with citation.
