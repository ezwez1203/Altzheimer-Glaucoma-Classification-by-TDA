#!/bin/bash
# run_all.sh - Run complete VCP pipeline (preprocess → train → inference → organize → evaluate)
# Just run this and go to sleep!
#
# Usage: ./run_all.sh
#        nohup ./run_all.sh > training.log 2>&1 &   # Run in background

# Don't exit on error - continue with remaining steps
# set -e

echo "=========================================="
echo "VCP Complete Pipeline"
echo "Started at: $(date)"
echo "=========================================="

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate bob_env1

cd "$(dirname "$0")"  # Change to script directory

# Step 1: Preprocessing
echo ""
echo "[1/5] Preprocessing..."
./01_preprocess.sh || echo "Warning: Preprocessing had errors, continuing..."

# Step 2: Training
echo ""
echo "[2/5] Training (this will take a while)..."
./02_train.sh || echo "Warning: Training had errors, continuing..."

# Step 3: Inference
echo ""
echo "[3/5] Running inference..."
./03_inference.sh || echo "Warning: Inference had errors, continuing..."

# Step 4: Organize results
echo ""
echo "[4/5] Organizing output files..."
./04_organize_processed_data.sh || echo "Warning: Organizing had errors, continuing..."

# Step 5: Evaluate accuracy
echo ""
echo "[5/5] Evaluating accuracy..."
./05_evaluate.sh || echo "Warning: Evaluation had errors, continuing..."

echo ""
echo "=========================================="
echo "VCP Pipeline Complete!"
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - checkpoints/     : Trained models"
echo "  - results/DRIVE/   : DRIVE predictions"
echo "  - results/IOSTAR/  : IOSTAR predictions"
echo "  - results/logs/    : Organized by type"
echo ""
echo "Check accuracy metrics above!"
