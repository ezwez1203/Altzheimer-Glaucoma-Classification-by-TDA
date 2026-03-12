#!/bin/bash
# 05_evaluate.sh - Evaluate AI accuracy against ground truth
# Run this AFTER inference to check model performance

echo "=== Step 5: Accuracy Evaluation ==="

python evaluate.py

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: results/evaluation_results.txt"
