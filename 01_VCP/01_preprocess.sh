#!/bin/bash
# 01_preprocess.sh - OPTIONAL: Generate thickness/orientation maps from ground truth
# Run this BEFORE training if you want proper connectivity network training

echo "=== Step 1: Preprocessing (Optional) ==="
echo "Generating thickness and orientation maps from ground truth..."

# DRIVE training set
echo "Processing DRIVE training set..."
python -m src.preprocessing data/DRIVE/training/1st_manual processed_data/DRIVE_training

# IOSTAR training set
echo "Processing IOSTAR training set..."
python -m src.preprocessing data/IOSTAR/training/1st_manual processed_data/IOSTAR_training

# Optional: DRIVE test set
# python -m src.preprocessing data/DRIVE/test/1st_manual processed_data/DRIVE_test

echo "Done! Maps saved to processed_data/"
