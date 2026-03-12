# Topology-Aware Retinal Artery-Vein Classification

Retinal vessel segmentation and artery-vein classification using **Vascular Connectivity Prediction (VCP)**.

Based on: *"Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"*

---

## Overview

Three-stage training pipeline:

1. **Optic Disc Segmentation** — VGG16 encoder, BCE + Dice loss
2. **Multi-task Network** — Joint vessel segmentation + AV classification (artery / vein / background)
3. **Connectivity Network** — Two-step training; predicts whether pairs of vessel pixels belong to the same vascular tree

Final AV classification is determined by graph-based tree tracing using the connectivity predictions.

---

## Architecture

| Model | Backbone | Output |
|-------|----------|--------|
| `OpticDiscSegmentationNetwork` | VGG16 | Binary OD mask |
| `MultiTaskNetwork` | VGG16 (default) or Attention U-Net++ | Vessel mask + AV map |
| `FullConnectivityPipeline` | VGG16 (default) or Attention U-Net++ | Thickness / Orientation / Connectivity |

Default backbone is **VGG16** (`USE_ATTENTION_UNET = False` in `config.py`).
To switch to Attention U-Net++, set `USE_ATTENTION_UNET = True`.

---

## Installation

```bash
conda create -n vcp python=3.9 -y
conda activate vcp
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
scipy>=1.7.0
Pillow>=8.0.0
scikit-image>=0.18.0
opencv-python>=4.5.0
tqdm>=4.60.0
tensorboard>=2.5.0
matplotlib>=3.4.0
```

---

## Quick Start

```bash
# 1. Preprocess — generate thickness/orientation maps (parallelized)
bash 01_preprocess.sh

# 2. Train
bash 02_train.sh
# or individually:
python train.py --dataset DRIVE --task od
python train.py --dataset DRIVE --task multitask --epochs 100
python train.py --dataset DRIVE --task connectivity --epochs 100

# 3. Inference
bash 03_inference.sh
# or:
python inference.py --dataset DRIVE

# 4. Evaluate
bash 05_evaluate.sh
# or:
python evaluate.py --dataset DRIVE
```

---

## Dataset Structure

```
data/
├── DRIVE/
│   ├── training/
│   │   ├── images/        RGB fundus images (20)
│   │   ├── 1st_manual/    vessel GT masks
│   │   ├── mask/          FOV masks
│   │   ├── av/            AV labels (red=artery, blue=vein)
│   │   └── od/            optic disc masks
│   └── test/
│       └── ...            (20 images, no AV labels)
└── IOSTAR/
    └── training/
        ├── images/
        ├── 1st_manual/
        ├── mask/
        └── od/
```

---

## Logging

Each training run writes:

- **TXT log** — `results/logs/{task}_{dataset}_{timestamp}.txt`
  Columns: `epoch / train_loss / val_loss / f1_artery / f1_vein / f1_macro` (multitask)
- **TensorBoard** — `results/logs/{task}_{dataset}_{timestamp}/`
  Scalars: `Loss/train`, `Loss/val`, `F1/artery`, `F1/vein`, `F1/macro`

```bash
tensorboard --logdir results/logs
```

---

## Checkpoints

```
checkpoints/
├── od_segm_{dataset}_best.pth
├── multitask_{dataset}_best.pth
├── connectivity_{dataset}_step1.pth
└── connectivity_{dataset}_best.pth
```

---

## Key Configuration (`config.py`)

| Parameter | Value | Note |
|-----------|-------|------|
| `IMAGE_SIZE` | (384, 384) | Reduced for 8 GB GPU |
| `BATCH_SIZE` | 1 | RTX 4060 limit |
| `USE_PRETRAINED_VGG` | True | ImageNet weights |
| `USE_ATTENTION_UNET` | **False** | VGG16 backbone (set True for Attention U-Net++) |
| `VAL_SPLIT` | 0.1 | 10 % of training set used for validation |
| `LAMBDA_SEGM_DRIVE` | 1.0 | Vessel loss weight (DRIVE) |
| `LAMBDA_AV_DRIVE` | 10.0 | AV loss weight (DRIVE) |

---

## Author

**Dohyun Hwang** — ezwez1467@yonsei.ac.kr

## License

For research purposes only.
