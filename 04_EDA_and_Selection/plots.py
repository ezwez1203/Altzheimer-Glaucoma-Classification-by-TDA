"""
plots.py — Academic-quality visualizations for the EDA report.
"""

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from config import PLOTS_DIR, FIGURE_DPI, FIGURE_FORMAT, P_VALUE_THRESHOLD


def _setup_style():
    """Configure seaborn/matplotlib for academic papers."""
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.0,
        "grid.alpha": 0.4,
    })


def generate_plots(
    df: pd.DataFrame,
    feature_cols: List[str],
    stats_df: pd.DataFrame,
    label_col: str = "label",
) -> None:
    """
    Generate and save all academic figures.

    1. Per-feature boxplots with swarmplot overlay
    2. Correlation heatmap

    Args:
        df:           merged DataFrame with features and labels.
        feature_cols: list of numeric feature column names.
        stats_df:     statistical test results (for annotation).
        label_col:    class label column.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    _setup_style()

    # filter constant features once
    var_cols = [c for c in feature_cols if df[c].std() > 0]

    _plot_boxplots(df, var_cols, stats_df, label_col)
    _plot_correlation_heatmap(df, var_cols)

    print(f"  Plots saved to {PLOTS_DIR}/")


def _plot_boxplots(
    df: pd.DataFrame,
    feature_cols: List[str],
    stats_df: pd.DataFrame,
    label_col: str,
) -> None:
    """Boxplot + swarmplot for each feature, grouped by class."""
    # build p-value lookup
    p_lookup = dict(zip(stats_df["feature"], stats_df["p_value"]))

    plot_cols = feature_cols
    if not plot_cols:
        print("  [WARN] no variable features to plot")
        return

    n_features = len(plot_cols)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    palette = {"Normal": "#4C72B0", "Glaucoma": "#DD8452", "Alzheimer": "#C44E52"}

    for i, feat in enumerate(plot_cols):
        ax = axes[i]

        sns.boxplot(
            data=df, x=label_col, y=feat, hue=label_col, ax=ax,
            palette=palette, width=0.5, linewidth=1.2,
            fliersize=0, legend=False,
        )

        # swarmplot overlay — skip if too many points
        if len(df) <= 100:
            sns.swarmplot(
                data=df, x=label_col, y=feat, hue=label_col, ax=ax,
                palette={"Normal": "0.25", "Glaucoma": "0.25", "Alzheimer": "0.25"},
                size=3, alpha=0.6, legend=False,
            )

        p_val = p_lookup.get(feat, 1.0)
        sig_marker = " *" if p_val < P_VALUE_THRESHOLD else ""
        ax.set_title(f"{feat}\n(p={p_val:.4f}{sig_marker})", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Clinical Class", fontsize=13, y=1.01)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f"boxplots.{FIGURE_FORMAT}")
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    """Pearson correlation heatmap among all features."""
    var_cols = feature_cols
    if len(var_cols) < 2:
        print("  [WARN] fewer than 2 variable features — skipping heatmap")
        return

    corr = df[var_cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(var_cols) * 0.9), max(6, len(var_cols) * 0.7)))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr, mask=mask, ax=ax,
        annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )

    ax.set_title("Feature Correlation Matrix", fontsize=13, pad=12)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f"correlation_heatmap.{FIGURE_FORMAT}")
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
