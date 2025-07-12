from pathlib import Path

DATA_DIR = Path("dataset/coco2017")
TRAIN_SET = DATA_DIR / "coco2017/train2017"
VAL_SET = DATA_DIR / "coco2017/val2017"
TRAIN_ANNOTATIONS = DATA_DIR / "annotations/instances_train2017.json"
VAL_ANNOTATIONS = DATA_DIR / "annotations/instances_val2017.json"
WEIGHTS_PATH = Path("weights")

BATCH_SIZE = 4
NUM_CLASSES = 80

MODEL = {
    "PARAMS": {
        "efficientdet-d0": {
            "phi": 0,
            "w_bifpn": 64,
            "d_bifpn": 2,
            "d_class": 3
        }
    },
    "URL": {
        "efficientnet-b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth"
    }
}
