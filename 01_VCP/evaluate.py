#!/usr/bin/env python3
"""
Accuracy Evaluation for VCP Project

Compares AI predictions with ground truth AV labels.
Calculates: Accuracy, Sensitivity, Specificity, Precision
Outputs results to: results/evaluation_results.txt
"""

import os
import sys
import numpy as np
from PIL import Image
from glob import glob
from datetime import datetime

def load_av_image(path):
    """Load AV image and return class labels (0=bg, 1=artery, 2=vein)"""
    img = np.array(Image.open(path).convert('RGB'))
    labels = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Artery = Red (R > 200, G < 100, B < 100)
    artery_mask = (img[:,:,0] > 200) & (img[:,:,1] < 100) & (img[:,:,2] < 100)
    labels[artery_mask] = 1
    
    # Vein = Blue (B > 200, R < 100, G < 100)  
    vein_mask = (img[:,:,2] > 200) & (img[:,:,0] < 100) & (img[:,:,1] < 100)
    labels[vein_mask] = 2
    
    return labels

def evaluate_dataset(pred_dir, gt_dir, name, output_file):
    """Evaluate predictions against ground truth"""
    if not os.path.exists(pred_dir) or not os.path.exists(gt_dir):
        output_file.write(f"\n  Skipping {name}: directories not found\n")
        return
    
    pred_files = glob(os.path.join(pred_dir, "*_av_treewise.png"))
    if not pred_files:
        output_file.write(f"\n  No predictions found in {pred_dir}\n")
        return
    
    gt_files = sorted(glob(os.path.join(gt_dir, "*")))
    
    total_pixels = 0
    correct_pixels = 0
    tp_a, fp_a, fn_a = 0, 0, 0
    tp_v, fp_v, fn_v = 0, 0, 0
    evaluated_count = 0
    
    for pred_path in sorted(pred_files):
        basename = os.path.basename(pred_path)
        prefix = basename[:2]
        
        gt_path = None
        for gf in gt_files:
            if prefix in os.path.basename(gf):
                gt_path = gf
                break
        
        if not gt_path:
            continue
        
        pred = load_av_image(pred_path)
        gt = load_av_image(gt_path)
        
        # Resize if needed
        if pred.shape != gt.shape:
            gt_img = Image.fromarray(gt)
            gt_img = gt_img.resize((pred.shape[1], pred.shape[0]), Image.NEAREST)
            gt = np.array(gt_img)
        
        # Vessel mask (artery or vein in GT)
        vessel_mask = gt > 0
        total_pixels += np.sum(vessel_mask)
        correct_pixels += np.sum((pred == gt) & vessel_mask)
        
        # Artery
        tp_a += np.sum((pred == 1) & (gt == 1))
        fp_a += np.sum((pred == 1) & (gt != 1))
        fn_a += np.sum((pred != 1) & (gt == 1))
        
        # Vein
        tp_v += np.sum((pred == 2) & (gt == 2))
        fp_v += np.sum((pred == 2) & (gt != 2))
        fn_v += np.sum((pred != 2) & (gt == 2))
        
        evaluated_count += 1
    
    # Calculate metrics
    accuracy = 100 * correct_pixels / total_pixels if total_pixels > 0 else 0
    sens_a = 100 * tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0
    sens_v = 100 * tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0
    prec_a = 100 * tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0
    prec_v = 100 * tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0
    spec_a = 100 * (total_pixels - tp_a - fp_a - fn_a) / (total_pixels - tp_a - fn_a) if (total_pixels - tp_a - fn_a) > 0 else 0
    spec_v = 100 * (total_pixels - tp_v - fp_v - fn_v) / (total_pixels - tp_v - fn_v) if (total_pixels - tp_v - fn_v) > 0 else 0

    output_file.write("\n")
    output_file.write("=" * 60 + "\n")
    output_file.write(f" Dataset: {name}\n")
    output_file.write("=" * 60 + "\n")
    output_file.write(f" Images evaluated: {evaluated_count}\n")
    output_file.write(f" OVERALL ACCURACY: {accuracy:.2f}%\n")
    output_file.write(f" Total vessel pixels: {total_pixels}\n")
    output_file.write("\n")
    output_file.write(f" {'Metric':<15} {'Artery':>12} {'Vein':>12}\n")
    output_file.write(f" {'-'*15} {'-'*12} {'-'*12}\n")
    output_file.write(f" {'Sensitivity':<15} {sens_a:>10.2f}% {sens_v:>10.2f}%\n")
    output_file.write(f" {'Specificity':<15} {spec_a:>10.2f}% {spec_v:>10.2f}%\n")
    output_file.write(f" {'Precision':<15} {prec_a:>10.2f}% {prec_v:>10.2f}%\n")
    output_file.write("\n")


def main():
    output_path = "results/evaluation_results.txt"
    os.makedirs("results", exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("╔══════════════════════════════════════════════════════════════╗\n")
        f.write("║           VCP Accuracy Evaluation Results                    ║\n")
        f.write("╚══════════════════════════════════════════════════════════════╝\n")
        f.write(f"\nEvaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        evaluate_dataset("results/DRIVE", "data/DRIVE/test/av", "DRIVE", f)
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(" Evaluation Complete!\n")
        f.write("=" * 60 + "\n")
    
    print(f"Evaluation results saved to: {output_path}")
    
    # Also print to console
    with open(output_path, "r") as f:
        print(f.read())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
