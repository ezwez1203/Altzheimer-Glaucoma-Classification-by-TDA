#!/bin/bash
# 02_train.sh - Train all networks
# Run this AFTER preprocessing (optional) to train the models

echo "=== Step 2: Training ==="

# Train all networks on COMBINED dataset (DRIVE + IOSTAR)
python train.py --dataset COMBINED --task all

# Or train on specific datasets:
# python train.py --dataset DRIVE --task all
# python train.py --dataset IOSTAR --task all

# Or train individual networks:
# python train.py --dataset COMBINED --task od --epochs 50
# python train.py --dataset COMBINED --task multitask --epochs 100
# python train.py --dataset COMBINED --task connectivity --epochs 100

echo "Done! Checkpoints saved to checkpoints/"
