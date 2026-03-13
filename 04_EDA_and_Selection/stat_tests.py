"""
statistics.py — Statistical significance testing across clinical classes.

Supports:
  - 3-class testing (Normal / Alzheimer / Glaucoma): ANOVA or Kruskal-Wallis
  - 2-class testing (binary): t-test or Mann-Whitney U
"""

from typing import List

import pandas as pd
from scipy.stats import kruskal, shapiro, f_oneway, ttest_ind, mannwhitneyu

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


def run_binary_tests(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    group_a_labels: List[str] = None,
    group_b_labels: List[str] = None,
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
) -> pd.DataFrame:
    """
    Two-group statistical testing for stage-specific feature selection.

    Strategy:
        1. Shapiro-Wilk normality check on each group.
        2. Both normal → independent t-test (Welch's).
        3. Otherwise → Mann-Whitney U test.

    Args:
        df:             DataFrame with features and labels.
        feature_cols:   numeric feature column names.
        label_col:      class label column.
        group_a_labels: label values for group A (e.g., ["Normal"]).
        group_b_labels: label values for group B (e.g., ["Alzheimer", "Glaucoma"]).
        group_a_name:   display name for group A.
        group_b_name:   display name for group B.

    Returns:
        DataFrame with feature, test_used, statistic, p_value, significant.
    """
    mask_a = df[label_col].isin(group_a_labels)
    mask_b = df[label_col].isin(group_b_labels)

    results = []
    for feat in feature_cols:
        ga = df.loc[mask_a, feat].dropna().values
        gb = df.loc[mask_b, feat].dropna().values

        # skip zero-variance features
        combined = pd.concat([pd.Series(ga), pd.Series(gb)])
        if combined.std() == 0:
            results.append({
                "feature": feat,
                "test_used": "skipped (zero variance)",
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "group_a_mean": 0.0,
                "group_b_mean": 0.0,
            })
            continue

        if len(ga) < 3 or len(gb) < 3:
            results.append({
                "feature": feat,
                "test_used": "skipped (too few samples)",
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "group_a_mean": float(ga.mean()) if len(ga) > 0 else 0.0,
                "group_b_mean": float(gb.mean()) if len(gb) > 0 else 0.0,
            })
            continue

        # normality check
        _, sw_pa = shapiro(ga)
        _, sw_pb = shapiro(gb)
        both_normal = sw_pa >= 0.05 and sw_pb >= 0.05

        if both_normal:
            stat, p_val = ttest_ind(ga, gb, equal_var=False)  # Welch's t-test
            test_name = "Welch t-test"
        else:
            stat, p_val = mannwhitneyu(ga, gb, alternative="two-sided")
            test_name = "Mann-Whitney U"

        results.append({
            "feature": feat,
            "test_used": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": p_val < P_VALUE_THRESHOLD,
            "group_a_mean": round(float(ga.mean()), 4),
            "group_b_mean": round(float(gb.mean()), 4),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("p_value").reset_index(drop=True)
    return results_df
