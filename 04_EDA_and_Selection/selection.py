"""
selection.py — Feature selection based on statistical significance.
"""

from typing import List

import pandas as pd

from config import P_VALUE_THRESHOLD, SELECTED_CSV


def select_features(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    p_threshold: float = P_VALUE_THRESHOLD,
) -> pd.DataFrame:
    """
    Retain only features that are statistically significant (p < threshold).

    If no features pass the threshold, falls back to retaining all features
    (with a warning) so the downstream SVM stage still has data to work with.

    Args:
        df:          merged DataFrame with all features and label.
        stats_df:    statistical results with 'feature' and 'p_value' columns.
        p_threshold: significance cutoff.

    Returns:
        DataFrame with subject_id, dataset, label, and selected features.
    """
    sig_features = stats_df.loc[stats_df["p_value"] < p_threshold, "feature"].tolist()

    if not sig_features:
        print(f"  [WARN] no features with p < {p_threshold}")
        print("         falling back to all non-constant features")
        # keep features that have any variance
        all_feats = stats_df.loc[stats_df["test_used"] != "skipped (zero variance)", "feature"].tolist()
        sig_features = all_feats

    keep_cols = ["subject_id", "dataset", "label"] + sig_features
    selected = df[keep_cols].copy()

    return selected


def save_selected(selected: pd.DataFrame, output_path: str = SELECTED_CSV) -> None:
    """Write selected features to CSV."""
    selected.to_csv(output_path, index=False)
    n_feats = len(selected.columns) - 3  # minus id, dataset, label
    print(f"  Selected {n_feats} features -> {output_path}")
