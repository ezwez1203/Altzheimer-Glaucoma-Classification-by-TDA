#!/bin/bash
# 03_inference.sh - Run inference on test images
# Run this AFTER training to generate predictions

echo "=== Step 3: Inference ==="

# Note: Checkpoints are named with COMBINED since we trained on combined dataset
# We need to specify the checkpoint paths explicitly

CHECKPOINT_DIR="checkpoints"
OD_CHECKPOINT="${CHECKPOINT_DIR}/od_segm_DRIVE_best.pth"
MULTITASK_CHECKPOINT="${CHECKPOINT_DIR}/multitask_COMBINED_best.pth"
CONNECTIVITY_CHECKPOINT="${CHECKPOINT_DIR}/connectivity_COMBINED_best.pth"

# Verify checkpoints exist
if [ ! -f "$MULTITASK_CHECKPOINT" ]; then
    echo "Error: Multi-task checkpoint not found: $MULTITASK_CHECKPOINT"
    echo "Please run training first: ./02_train.sh"
    exit 1
fi

# Run inference on all DRIVE test images
echo "Running inference on DRIVE test images..."
python inference.py data/DRIVE/test/images/ \
    --output results/DRIVE/ \
    --od-checkpoint "$OD_CHECKPOINT" \
    --multitask-checkpoint "$MULTITASK_CHECKPOINT" \
    --connectivity-checkpoint "$CONNECTIVITY_CHECKPOINT"

# Run inference on IOSTAR training images (no separate test set)
echo "Running inference on IOSTAR images..."
python inference.py data/IOSTAR/training/images/ \
    --output results/IOSTAR/ \
    --od-checkpoint "$OD_CHECKPOINT" \
    --multitask-checkpoint "$MULTITASK_CHECKPOINT" \
    --connectivity-checkpoint "$CONNECTIVITY_CHECKPOINT"

echo ""
echo "Done! Results saved to results/"
echo "Output files:"
echo "  - *_vessel.png      : Vessel segmentation"
echo "  - *_thickness.png   : Vessel thickness map"
echo "  - *_orientation.png : Vessel orientation map"
echo "  - *_av_pixelwise.png: Pixelwise AV classification"
echo "  - *_av_treewise.png : Tree-wise AV classification"
echo "  - *_topology.png    : Topology visualization"
echo "  - *_od.png          : Optic disc segmentation"
