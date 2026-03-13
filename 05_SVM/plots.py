"""
plots.py — Academic-quality confusion matrix and ROC curve visualizations.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import RESULTS_DIR, FIGURE_DPI, FIGURE_FORMAT


def _setup_style():
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.0,
        "grid.alpha": 0.4,
    })


def plot_confusion_matrix(cm, class_names):
    """
    Normalized confusion matrix with raw-count annotations.

    Rows = true class, columns = predicted class.
    Cell color = row-normalized rate; small gray text shows (n=count).
    """
    _setup_style()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=1, linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.8, "label": "Rate"}, ax=ax,
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.75,
                f"(n={cm[i, j]})",
                ha="center", va="center", fontsize=7, color="gray",
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (K-Fold Aggregated)", fontsize=12, pad=10)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"confusion_matrix.{FIGURE_FORMAT}")
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to {out}")


def plot_roc_curves(roc_data, class_names):
    """
    Multi-class one-vs-rest ROC curves with per-class AUC and macro AUC.
    """
    _setup_style()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868", "#8172B2"]
    fig, ax = plt.subplots(figsize=(6, 5))

    for c in range(len(class_names)):
        if c not in roc_data:
            continue
        fpr = roc_data[c]["fpr"]
        tpr = roc_data[c]["tpr"]
        auc_val = roc_data[c]["auc"]
        ax.plot(fpr, tpr, color=colors[c % len(colors)], lw=1.8,
                label=f"{class_names[c]} (AUC = {auc_val:.3f})")

    macro_auc = roc_data.get("macro_auc", 0.0)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Chance")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"ROC Curves — One-vs-Rest  (Macro AUC = {macro_auc:.3f})",
        fontsize=11, pad=10,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"roc_curves.{FIGURE_FORMAT}")
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curves saved to {out}")
