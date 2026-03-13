"""
main.py — Entry point for EDA & Feature Selection.

Produces three sets of selected features:
  1. selected_features.csv        — 3-class (original, ANOVA/Kruskal-Wallis)
  2. selected_features_stage1.csv — Normal vs Disease (t-test/Mann-Whitney U)
  3. selected_features_stage2.csv — Alzheimer vs Glaucoma (t-test/Mann-Whitney U)

Usage:
    conda activate cuda_tda
    python main.py
"""

from config import (
    STATS_CSV, STATS_STAGE1_CSV, STATS_STAGE2_CSV,
    SELECTED_STAGE1_CSV, SELECTED_STAGE2_CSV,
)
from data_loader import load_and_merge_data, get_feature_columns
from stat_tests import run_statistical_tests, run_binary_tests
from plots import generate_plots
from selection import select_features, save_selected


def main():
    # step 1 — load and merge
    print("[1/6] Loading and merging data ...")
    df = load_and_merge_data()

    feature_cols = get_feature_columns(df)
    print(f"  Features: {feature_cols}")

    # step 2 — 3-class statistical testing (original)
    print("\n[2/6] 3-class statistical tests (Normal / Alzheimer / Glaucoma) ...")
    stats_df = run_statistical_tests(df, feature_cols)
    stats_df.to_csv(STATS_CSV, index=False)
    print(stats_df.to_string(index=False))

    # step 3 — Stage 1: Normal vs Disease (2-class)
    print("\n[3/6] Stage 1 tests: Normal vs Disease (AD + Glaucoma) ...")
    stats_s1 = run_binary_tests(
        df, feature_cols,
        group_a_labels=["Normal"],
        group_b_labels=["Alzheimer", "Glaucoma"],
        group_a_name="Normal",
        group_b_name="Disease",
    )
    stats_s1.to_csv(STATS_STAGE1_CSV, index=False)
    print(stats_s1.to_string(index=False))

    # step 4 — Stage 2: Alzheimer vs Glaucoma (2-class)
    print("\n[4/6] Stage 2 tests: Alzheimer vs Glaucoma ...")
    df_disease = df[df["label"].isin(["Alzheimer", "Glaucoma"])].copy()
    stats_s2 = run_binary_tests(
        df_disease, feature_cols,
        group_a_labels=["Alzheimer"],
        group_b_labels=["Glaucoma"],
        group_a_name="Alzheimer",
        group_b_name="Glaucoma",
    )
    stats_s2.to_csv(STATS_STAGE2_CSV, index=False)
    print(stats_s2.to_string(index=False))

    # step 5 — visualization
    print("\n[5/6] Generating plots ...")
    generate_plots(df, feature_cols, stats_df)

    # step 6 — feature selection (all three sets)
    print("\n[6/6] Selecting significant features ...")

    # 3-class (original)
    selected_3c = select_features(df, stats_df)
    save_selected(selected_3c)

    # Stage 1: Normal vs Disease
    selected_s1 = select_features(df, stats_s1)
    save_selected(selected_s1, output_path=SELECTED_STAGE1_CSV)

    # Stage 2: Alzheimer vs Glaucoma (only disease rows)
    selected_s2 = select_features(df_disease, stats_s2)
    save_selected(selected_s2, output_path=SELECTED_STAGE2_CSV)

    print("\nDone.")


if __name__ == "__main__":
    main()
