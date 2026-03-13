"""
main.py — Entry point for EDA & Feature Selection.

Usage:
    conda activate cuda_tda
    python main.py
"""

from config import STATS_CSV
from data_loader import load_and_merge_data, get_feature_columns
from stat_tests import run_statistical_tests
from plots import generate_plots
from selection import select_features, save_selected


def main():
    # step 1 — load and merge
    print("[1/4] Loading and merging data ...")
    df = load_and_merge_data()

    feature_cols = get_feature_columns(df)
    print(f"  Features: {feature_cols}")

    # step 2 — statistical testing
    print("\n[2/4] Running statistical tests ...")
    stats_df = run_statistical_tests(df, feature_cols)
    stats_df.to_csv(STATS_CSV, index=False)
    print(f"  Results saved to {STATS_CSV}")
    print(stats_df.to_string(index=False))

    # step 3 — visualization
    print("\n[3/4] Generating plots ...")
    generate_plots(df, feature_cols, stats_df)

    # step 4 — feature selection
    print("\n[4/4] Selecting significant features ...")
    selected = select_features(df, stats_df)
    save_selected(selected)

    print("\nDone.")


if __name__ == "__main__":
    main()
