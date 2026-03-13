"""
statistics.py — Statistical significance testing across clinical classes.
"""

from typing import List

import pandas as pd
from scipy.stats import kruskal, shapiro, f_oneway

from config import P_VALUE_THRESHOLD


def run_statistical_tests(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
) -> pd.DataFrame:
    """
    For each feature, test whether it differs significantly across classes.

    Strategy:
        1. Check normality with Shapiro-Wilk on each group.
        2. If all groups pass normality (p > 0.05): one-way ANOVA.
        3. Otherwise: Kruskal-Wallis H-test.

    Args:
        df:           merged DataFrame with features and labels.
        feature_cols: list of numeric feature column names.
        label_col:    column name for the class label.

    Returns:
        DataFrame with columns:
            feature, test_used, statistic, p_value, significant
    """
    classes = df[label_col].unique()
    results = []

    for feat in feature_cols:
        groups = [df.loc[df[label_col] == c, feat].dropna().values for c in classes]

        # skip features with zero variance in all groups
        all_vals = pd.concat([pd.Series(g) for g in groups])
        if all_vals.std() == 0:
            results.append({
                "feature": feat,
                "test_used": "skipped (zero variance)",
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
            })
            continue

        # skip groups with fewer than 3 samples
        valid_groups = [g for g in groups if len(g) >= 3]
        if len(valid_groups) < 2:
            results.append({
                "feature": feat,
                "test_used": "skipped (too few samples)",
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
            })
            continue

        # normality check per group
        all_normal = True
        for g in valid_groups:
            if len(g) < 3:
                all_normal = False
                break
            _, sw_p = shapiro(g)
            if sw_p < 0.05:
                all_normal = False
                break

        # choose test
        if all_normal and len(valid_groups) >= 2:
            stat, p_val = f_oneway(*valid_groups)
            test_name = "ANOVA"
        else:
            stat, p_val = kruskal(*valid_groups)
            test_name = "Kruskal-Wallis"

        results.append({
            "feature": feat,
            "test_used": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": p_val < P_VALUE_THRESHOLD,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("p_value").reset_index(drop=True)
    return results_df
