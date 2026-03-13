"""
data_loader.py — Load, merge, and label the feature CSVs.
"""

import os

import numpy as np
import pandas as pd

from config import TRACK_A_CSV, TRACK_B_CSV, GRAPH_MACRO_CSV, LABELS_CSV, ID_COLS


def load_and_merge_data() -> pd.DataFrame:
    """
    Load Track A and Track B feature CSVs, merge on subject_id + dataset,
    then join with clinical labels.

    If labels.csv is missing, generates mock labels for testing:
        - DRIVE subjects → randomly assigned Normal / Glaucoma
        - IOSTAR subjects → randomly assigned Normal / Alzheimer

    Returns:
        Merged DataFrame with all features and a 'label' column.
    """
    # load feature CSVs
    df_a = pd.read_csv(TRACK_A_CSV)
    df_b = pd.read_csv(TRACK_B_CSV)

    print(f"  Track A: {len(df_a)} rows, {len(df_a.columns)} cols")
    print(f"  Track B: {len(df_b)} rows, {len(df_b.columns)} cols")

    # merge on subject_id + dataset
    df = pd.merge(df_a, df_b, on=ID_COLS, how="outer")
    print(f"  Merged:  {len(df)} rows, {len(df.columns)} cols")

    # merge graph macro features (if available)
    if os.path.exists(GRAPH_MACRO_CSV):
        df_g = pd.read_csv(GRAPH_MACRO_CSV)
        print(f"  Graph:   {len(df_g)} rows, {len(df_g.columns)} cols")
        df = pd.merge(df, df_g, on=ID_COLS, how="left")
        print(f"  + Graph: {len(df)} rows, {len(df.columns)} cols")

    # load or generate labels
    if os.path.exists(LABELS_CSV):
        labels = pd.read_csv(LABELS_CSV)
        print(f"  Labels loaded from {LABELS_CSV}")
    else:
        print(f"  [WARN] {LABELS_CSV} not found — generating mock labels for testing")
        labels = _generate_mock_labels(df)

    df = pd.merge(df, labels, on="subject_id", how="left")

    # fill any missing labels
    if df["label"].isna().any():
        n_missing = df["label"].isna().sum()
        print(f"  [WARN] {n_missing} subjects without labels, assigning 'Normal'")
        df["label"] = df["label"].fillna("Normal")

    print(f"  Final:   {len(df)} rows, classes: {df['label'].value_counts().to_dict()}")
    return df


def _generate_mock_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create mock clinical labels for testing when labels.csv is absent.

    Assignment:
        DRIVE  → 50% Normal, 50% Glaucoma
        IOSTAR → 50% Normal, 50% Alzheimer
    """
    rng = np.random.RandomState(42)
    rows = []

    for _, row in df.iterrows():
        sid = row["subject_id"]
        ds = row["dataset"]

        if ds == "DRIVE":
            label = rng.choice(["Normal", "Glaucoma"])
        else:
            label = rng.choice(["Normal", "Alzheimer"])

        rows.append({"subject_id": sid, "label": label})

    labels_df = pd.DataFrame(rows)

    # save for reproducibility
    labels_df.to_csv(LABELS_CSV, index=False)
    print(f"  Mock labels saved to {LABELS_CSV}")

    return labels_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return only numeric feature column names (exclude IDs and label)."""
    exclude = set(ID_COLS + ["label"])
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
