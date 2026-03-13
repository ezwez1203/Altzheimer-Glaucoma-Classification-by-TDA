"""
svm_classifier.py — Feature-Specific Hierarchical 2-stage SVM.

Architecture:
  Stage 1: Normal(23) vs Disease(27)
    Features: avg_edge_thickness (p=0.023), vessel_density (p=0.051)
    → macroscopic graph features that capture global vascular changes

  Stage 2: Alzheimer(20) vs Glaucoma(7)
    Features: fractal_dimension (p=0.007), b0_sum_lifespan (p=0.013)
    → TDA features that capture fine structural differences

Pipeline: StandardScaler → SVC (scaling inside CV, no data leakage)
Evaluation: RepeatedStratifiedKFold (3×5 = 15 evaluations) on N=50

Usage:
    cd 05_SVM
    python svm_classifier.py
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Master CSV (3-class selected features — superset of both stage feature sets)
DATA_CSV     = PROJECT_ROOT / "04_EDA_and_Selection" / "selected_features.csv"
RESULTS_DIR  = SCRIPT_DIR / "results"
MODEL_PATH   = RESULTS_DIR / "best_svm_model.pkl"
METRICS_CSV  = RESULTS_DIR / "fold_metrics.csv"

ID_COLS   = ["subject_id", "dataset"]
LABEL_COL = "label"

# ── Stage-specific feature sets ──
# Stage 1 (Normal vs Disease): graph macro + TDA (all p < 0.1)
STAGE1_FEATURES = ["avg_edge_thickness", "vessel_density", "fractal_dimension", "b0_sum_lifespan"]
# Stage 2 (AD vs Glaucoma): TDA topological features
STAGE2_FEATURES = ["fractal_dimension", "b0_sum_lifespan"]

N_SPLITS    = 5
N_REPEATS   = 3
RANDOM_SEED = 42

PARAM_GRID = [
    {
        "svc__kernel": ["linear"],
        "svc__C": [0.01, 0.1, 1, 10, 100],
    },
    {
        "svc__kernel": ["rbf"],
        "svc__C": [0.01, 0.1, 1, 10, 100],
        "svc__gamma": ["scale", "auto", 0.1, 0.01, 0.001],
    },
]

FIGURE_DPI    = 300
FIGURE_FORMAT = "png"

# Label mapping (after LabelEncoder): Alzheimer=0, Glaucoma=1, Normal=2
DISEASE_CLASSES = {0, 1}
NORMAL_CLASS = 2


# ──────────────────────────────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────────────────────────────

def load_data(csv_path=DATA_CSV):
    """Load master CSV and create stage-specific feature arrays.

    Returns:
        X_s1:          Stage 1 features (N × 2)
        X_s2:          Stage 2 features (N × 2)
        y:             encoded labels
        class_names:   list of class strings
        le:            fitted LabelEncoder
    """
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples from {csv_path}")

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].values)
    class_names = list(le.classes_)

    for feat in STAGE1_FEATURES + STAGE2_FEATURES:
        assert feat in df.columns, f"Feature '{feat}' not in CSV: {list(df.columns)}"

    X_s1 = df[STAGE1_FEATURES].values.astype(np.float64)
    X_s2 = df[STAGE2_FEATURES].values.astype(np.float64)

    counts = dict(zip(class_names, [int(c) for c in np.bincount(y)]))
    print(f"  Stage 1 features ({len(STAGE1_FEATURES)}): {STAGE1_FEATURES}")
    print(f"  Stage 2 features ({len(STAGE2_FEATURES)}): {STAGE2_FEATURES}")
    print(f"  Classes     : {class_names}")
    print(f"  Distribution: {counts}")

    return X_s1, X_s2, y, class_names, le


# ──────────────────────────────────────────────────────────────────────
#  Pipeline Builder
# ──────────────────────────────────────────────────────────────────────

def _build_pipeline():
    """Create a fresh Pipeline(StandardScaler → SVC)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])


def _fit_gridsearch(X_tr, y_tr, inner_k):
    """Fit GridSearchCV on given training data and return it."""
    gs = GridSearchCV(
        _build_pipeline(), PARAM_GRID,
        cv=StratifiedKFold(inner_k, shuffle=True, random_state=RANDOM_SEED),
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_tr, y_tr)
    return gs


# ──────────────────────────────────────────────────────────────────────
#  Hierarchical Prediction (feature-specific routing)
# ──────────────────────────────────────────────────────────────────────

def _hierarchical_predict(stage1_gs, stage2_gs, X_s1, X_s2, class_names):
    """
    Feature-specific hierarchical prediction.

    Stage 1: predict Normal(1) vs Disease(0) using X_s1
    Stage 2: for Disease predictions, predict AD(0) vs Glaucoma(1) using X_s2

    Returns:
        y_pred (3-class), y_prob (N × 3 probability matrix)
    """
    n = len(X_s1)
    n_classes = len(class_names)
    y_pred = np.full(n, NORMAL_CLASS, dtype=int)
    y_prob = np.zeros((n, n_classes))

    # Stage 1: Normal vs Disease
    s1_pred = stage1_gs.predict(X_s1)          # 0=Disease, 1=Normal
    s1_prob = stage1_gs.predict_proba(X_s1)    # [:, 0]=P(Disease), [:, 1]=P(Normal)

    normal_mask = s1_pred == 1
    disease_mask = s1_pred == 0

    # Normal predictions
    y_prob[normal_mask, NORMAL_CLASS] = s1_prob[normal_mask, 1]
    for dc in DISEASE_CLASSES:
        y_prob[normal_mask, dc] = s1_prob[normal_mask, 0] / len(DISEASE_CLASSES)

    # Stage 2: classify Disease samples
    if disease_mask.any():
        X_s2_disease = X_s2[disease_mask]
        s2_pred = stage2_gs.predict(X_s2_disease)
        s2_prob = stage2_gs.predict_proba(X_s2_disease)

        disease_indices = np.where(disease_mask)[0]
        disease_classes_sorted = sorted(DISEASE_CLASSES)

        for i, idx in enumerate(disease_indices):
            pred_s2 = s2_pred[i]
            y_pred[idx] = disease_classes_sorted[pred_s2]

            p_disease = s1_prob[idx, 0]
            for j, dc in enumerate(disease_classes_sorted):
                y_prob[idx, dc] = p_disease * s2_prob[i, j]
            y_prob[idx, NORMAL_CLASS] = s1_prob[idx, 1]

    return y_pred, y_prob


# ──────────────────────────────────────────────────────────────────────
#  Core Training — Feature-Specific Hierarchical + Repeated K-Fold
# ──────────────────────────────────────────────────────────────────────

def _macro_specificity(y_true, y_pred, n_classes):
    """Macro-averaged specificity: mean of per-class TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specs = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        denom = tn + fp
        specs.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(specs))


def _compute_roc(y_true, y_prob, n_classes):
    """Per-class OVR ROC curves + macro AUROC."""
    y_bin = label_binarize(y_true, classes=range(n_classes))
    result = {}
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        result[c] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    try:
        result["macro_auc"] = roc_auc_score(
            y_bin, y_prob, average="macro", multi_class="ovr",
        )
    except ValueError:
        result["macro_auc"] = float(
            np.mean([result[c]["auc"] for c in range(n_classes)])
        )
    return result


def train_and_evaluate(X_s1, X_s2, y, class_names):
    """Feature-specific Hierarchical SVM with Repeated Stratified K-Fold CV."""
    n_classes = len(class_names)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED,
    )

    fold_metrics = []
    first_pass_y_true, first_pass_y_pred = [], []
    first_pass_y_prob = np.zeros((len(y), n_classes))
    first_pass_counts = np.zeros(len(y), dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X_s1, y)):
        y_tr, y_te = y[train_idx], y[test_idx]

        # Feature-specific slicing
        X_s1_tr, X_s1_te = X_s1[train_idx], X_s1[test_idx]
        X_s2_tr, X_s2_te = X_s2[train_idx], X_s2[test_idx]

        # ── Stage 1: Normal vs Disease ──
        y_tr_s1 = np.where(np.isin(y_tr, list(DISEASE_CLASSES)), 0, 1)
        min_s1 = int(np.bincount(y_tr_s1).min())
        inner_k_s1 = max(2, min(3, min_s1))
        stage1_gs = _fit_gridsearch(X_s1_tr, y_tr_s1, inner_k_s1)

        # ── Stage 2: AD vs Glaucoma, Disease only ──
        disease_mask_tr = np.isin(y_tr, list(DISEASE_CLASSES))
        X_s2_tr_disease = X_s2_tr[disease_mask_tr]
        y_tr_s2 = y_tr[disease_mask_tr]
        min_s2 = int(np.bincount(y_tr_s2).min())
        inner_k_s2 = max(2, min(3, min_s2))
        stage2_gs = _fit_gridsearch(X_s2_tr_disease, y_tr_s2, inner_k_s2)

        # ── Per-stage evaluation ──
        y_te_s1 = np.where(np.isin(y_te, list(DISEASE_CLASSES)), 0, 1)
        s1_pred_te = stage1_gs.predict(X_s1_te)
        s1_acc = accuracy_score(y_te_s1, s1_pred_te)
        s1_f1 = f1_score(y_te_s1, s1_pred_te, average="macro", zero_division=0)

        disease_mask_te = np.isin(y_te, list(DISEASE_CLASSES))
        if disease_mask_te.any():
            X_s2_te_disease = X_s2_te[disease_mask_te]
            y_te_s2 = y_te[disease_mask_te]
            s2_pred_te = stage2_gs.predict(X_s2_te_disease)
            s2_acc = accuracy_score(y_te_s2, s2_pred_te)
            s2_f1 = f1_score(y_te_s2, s2_pred_te, average="macro", zero_division=0)
            s2_n = int(disease_mask_te.sum())
        else:
            s2_acc, s2_f1, s2_n = float("nan"), float("nan"), 0

        # ── Hierarchical prediction ──
        y_pred, y_prob = _hierarchical_predict(
            stage1_gs, stage2_gs, X_s1_te, X_s2_te, class_names,
        )

        first_pass_y_prob[test_idx] += y_prob
        first_pass_counts[test_idx] += 1
        if fold_i < N_SPLITS:
            first_pass_y_true.extend(y_te)
            first_pass_y_pred.extend(y_pred)

        acc  = accuracy_score(y_te, y_pred)
        f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)
        sens = recall_score(y_te, y_pred, average="macro", zero_division=0)
        spec = _macro_specificity(y_te, y_pred, n_classes)
        prec = precision_score(y_te, y_pred, average="macro", zero_division=0)

        s1_params = {k.replace("svc__", ""): v
                     for k, v in stage1_gs.best_params_.items()}
        s2_params = {k.replace("svc__", ""): v
                     for k, v in stage2_gs.best_params_.items()}

        repeat_num = fold_i // N_SPLITS + 1
        fold_num = fold_i % N_SPLITS + 1

        fold_metrics.append({
            "repeat": repeat_num,
            "fold": fold_num,
            "accuracy": acc,
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "f1_macro": f1,
            "s1_acc": s1_acc,
            "s1_f1": s1_f1,
            "s2_acc": s2_acc,
            "s2_f1": s2_f1,
            "s2_n": s2_n,
            "stage1_params": s1_params,
            "stage2_params": s2_params,
        })
        print(f"  Repeat {repeat_num} Fold {fold_num}/{N_SPLITS}  "
              f"3-class Acc={acc:.3f}  F1={f1:.3f}")
        print(f"    Stage1(Normal/Disease): Acc={s1_acc:.3f}  F1={s1_f1:.3f}  {s1_params}")
        print(f"    Stage2(AD/Glaucoma)   : Acc={s2_acc:.3f}  F1={s2_f1:.3f}  (n={s2_n})  {s2_params}")

    # ── Aggregate ──
    aggregate_cm = confusion_matrix(
        np.array(first_pass_y_true), np.array(first_pass_y_pred),
        labels=range(n_classes),
    )
    mask = first_pass_counts > 0
    first_pass_y_prob[mask] /= first_pass_counts[mask, np.newaxis]
    roc_data = _compute_roc(y, first_pass_y_prob, n_classes)

    # ── Final models on full dataset ──
    y_s1_full = np.where(np.isin(y, list(DISEASE_CLASSES)), 0, 1)
    min_s1_all = int(np.bincount(y_s1_full).min())
    inner_k_s1_all = max(2, min(N_SPLITS, min_s1_all))
    final_stage1 = _fit_gridsearch(X_s1, y_s1_full, inner_k_s1_all)

    disease_mask_full = np.isin(y, list(DISEASE_CLASSES))
    X_s2_disease_full = X_s2[disease_mask_full]
    y_s2_full = y[disease_mask_full]
    min_s2_all = int(np.bincount(y_s2_full).min())
    inner_k_s2_all = max(2, min(N_SPLITS, min_s2_all))
    final_stage2 = _fit_gridsearch(X_s2_disease_full, y_s2_full, inner_k_s2_all)

    # ── Summary ──
    metric_keys = ["accuracy", "sensitivity", "specificity", "precision", "f1_macro"]
    print(f"\n  === Feature-Specific Hierarchical SVM — {N_REPEATS}×{N_SPLITS}-Fold CV ===")
    print(f"\n  ── Stage 1: Normal vs Disease ({len(STAGE1_FEATURES)}f: {STAGE1_FEATURES}) ──")
    s1_accs = [m["s1_acc"] for m in fold_metrics]
    s1_f1s  = [m["s1_f1"] for m in fold_metrics]
    print(f"    {'accuracy':20s}: {np.mean(s1_accs):.4f} ± {np.std(s1_accs):.4f}")
    print(f"    {'f1_macro':20s}: {np.mean(s1_f1s):.4f} ± {np.std(s1_f1s):.4f}")

    print(f"\n  ── Stage 2: Alzheimer vs Glaucoma ({len(STAGE2_FEATURES)}f: {STAGE2_FEATURES}) ──")
    s2_accs = [m["s2_acc"] for m in fold_metrics if not np.isnan(m["s2_acc"])]
    s2_f1s  = [m["s2_f1"] for m in fold_metrics if not np.isnan(m["s2_f1"])]
    print(f"    {'accuracy':20s}: {np.mean(s2_accs):.4f} ± {np.std(s2_accs):.4f}")
    print(f"    {'f1_macro':20s}: {np.mean(s2_f1s):.4f} ± {np.std(s2_f1s):.4f}")

    print(f"\n  ── Final 3-class (hierarchical combined) ──")
    for k in metric_keys:
        vals = [m[k] for m in fold_metrics]
        print(f"    {k:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    macro_auc = roc_data.get("macro_auc", 0.0)
    print(f"    {'auroc_macro':20s}: {macro_auc:.4f}")

    s1_final = {k.replace("svc__", ""): v
                for k, v in final_stage1.best_params_.items()}
    s2_final = {k.replace("svc__", ""): v
                for k, v in final_stage2.best_params_.items()}
    print(f"  Final Stage1 params: {s1_final}")
    print(f"  Final Stage2 params: {s2_final}")

    best_models = {
        "stage1": final_stage1.best_estimator_,
        "stage2": final_stage2.best_estimator_,
    }

    return fold_metrics, aggregate_cm, roc_data, best_models


# ──────────────────────────────────────────────────────────────────────
#  Metrics CSV Export
# ──────────────────────────────────────────────────────────────────────

def save_fold_csv(fold_metrics):
    """Export per-fold metrics + aggregate (mean ± std) to CSV."""
    rows = []
    for m in fold_metrics:
        s1 = m["stage1_params"]
        s2 = m["stage2_params"]
        rows.append({
            "repeat": m["repeat"],
            "fold": m["fold"],
            "s1_acc": round(m["s1_acc"], 4),
            "s1_f1": round(m["s1_f1"], 4),
            "s2_acc": round(m["s2_acc"], 4) if not np.isnan(m["s2_acc"]) else "",
            "s2_f1": round(m["s2_f1"], 4) if not np.isnan(m["s2_f1"]) else "",
            "s2_n": m["s2_n"],
            "accuracy": round(m["accuracy"], 4),
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "precision": round(m["precision"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "s1_kernel": s1.get("kernel", ""),
            "s1_C": s1.get("C", ""),
            "s1_gamma": s1.get("gamma", ""),
            "s2_kernel": s2.get("kernel", ""),
            "s2_C": s2.get("C", ""),
            "s2_gamma": s2.get("gamma", ""),
        })

    metric_keys = ["accuracy", "sensitivity", "specificity", "precision", "f1_macro"]
    agg_mean = {k: round(np.mean([m[k] for m in fold_metrics]), 4)
                for k in metric_keys}
    agg_std = {k: round(np.std([m[k] for m in fold_metrics]), 4)
               for k in metric_keys}
    agg_mean["s1_acc"] = round(np.mean([m["s1_acc"] for m in fold_metrics]), 4)
    agg_mean["s1_f1"] = round(np.mean([m["s1_f1"] for m in fold_metrics]), 4)
    agg_std["s1_acc"] = round(np.std([m["s1_acc"] for m in fold_metrics]), 4)
    agg_std["s1_f1"] = round(np.std([m["s1_f1"] for m in fold_metrics]), 4)
    s2_valid = [m for m in fold_metrics if not np.isnan(m["s2_acc"])]
    agg_mean["s2_acc"] = round(np.mean([m["s2_acc"] for m in s2_valid]), 4)
    agg_mean["s2_f1"] = round(np.mean([m["s2_f1"] for m in s2_valid]), 4)
    agg_std["s2_acc"] = round(np.std([m["s2_acc"] for m in s2_valid]), 4)
    agg_std["s2_f1"] = round(np.std([m["s2_f1"] for m in s2_valid]), 4)

    rows.append({
        "repeat": "mean", "fold": "",
        **agg_mean, "s2_n": "",
        "s1_kernel": "", "s1_C": "", "s1_gamma": "",
        "s2_kernel": "", "s2_C": "", "s2_gamma": "",
    })
    rows.append({
        "repeat": "std", "fold": "",
        **agg_std, "s2_n": "",
        "s1_kernel": "", "s1_C": "", "s1_gamma": "",
        "s2_kernel": "", "s2_C": "", "s2_gamma": "",
    })

    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
    print(f"  Fold metrics saved to {METRICS_CSV}")


# ──────────────────────────────────────────────────────────────────────
#  Visualisation
# ──────────────────────────────────────────────────────────────────────

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
    """Row-normalised confusion matrix with raw-count annotations."""
    _setup_style()

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
                j + 0.5, i + 0.75, f"(n={cm[i, j]})",
                ha="center", va="center", fontsize=7, color="gray",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Feature-Specific Hierarchical SVM — Confusion Matrix",
                 fontsize=11, pad=10)

    fig.tight_layout()
    out = RESULTS_DIR / f"confusion_matrix.{FIGURE_FORMAT}"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to {out}")


def plot_roc_curves(roc_data, class_names):
    """Multi-class OVR ROC curves with per-class and macro AUC."""
    _setup_style()

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
    out = RESULTS_DIR / f"roc_curves.{FIGURE_FORMAT}"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curves saved to {out}")


def plot_decision_boundary(X_s1, X_s2, y, best_models, class_names):
    """Decision boundary plots. Stage 1 scatter only if >2D, Stage 2 full boundary."""
    _setup_style()

    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    markers = ["o", "s", "D"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Stage 1: scatter plot (top-2 features by importance for visualization)
    ax1 = axes[0]
    y_binary = np.where(np.isin(y, list(DISEASE_CLASSES)), 0, 1)

    # Use first 2 features for scatter visualization
    for cls, name, color in [(0, "Disease", "#C44E52"), (1, "Normal", "#4C72B0")]:
        mask_c = y_binary == cls
        ax1.scatter(X_s1[mask_c, 0], X_s1[mask_c, 1],
                    c=[color], marker=markers[cls],
                    s=55, edgecolors="white", linewidth=0.6,
                    label=f"{name} (n={mask_c.sum()})", zorder=3)
    ax1.set_xlabel(STAGE1_FEATURES[0])
    ax1.set_ylabel(STAGE1_FEATURES[1])
    n_feat = len(STAGE1_FEATURES)
    ax1.set_title(f"Stage 1: Normal vs Disease ({n_feat}D, showing top-2)", fontsize=11)
    ax1.legend(fontsize=9, framealpha=0.9)

    # Stage 2: full 2D decision boundary
    ax2 = axes[1]
    margin = 0.1
    x_min, x_max = X_s2[:, 0].min(), X_s2[:, 0].max()
    y_min, y_max = X_s2[:, 1].min(), X_s2[:, 1].max()
    x_range, y_range = x_max - x_min, y_max - y_min
    xx2, yy2 = np.meshgrid(
        np.linspace(x_min - margin * x_range, x_max + margin * x_range, 300),
        np.linspace(y_min - margin * y_range, y_max + margin * y_range, 300),
    )
    grid2 = np.c_[xx2.ravel(), yy2.ravel()]
    Z2 = best_models["stage2"].predict(grid2)
    disease_classes_sorted = sorted(DISEASE_CLASSES)
    Z2 = np.array([disease_classes_sorted[p] for p in Z2]).reshape(xx2.shape)

    cmap_bg = matplotlib.colors.ListedColormap(
        [c + "33" for c in colors[:len(class_names)]]
    )
    ax2.contourf(xx2, yy2, Z2, alpha=0.25, cmap=cmap_bg,
                 levels=np.arange(len(class_names) + 1) - 0.5)
    ax2.contour(xx2, yy2, Z2, colors="gray", linewidths=0.5, alpha=0.5)

    for c in range(len(class_names)):
        mask_c = y == c
        ax2.scatter(X_s2[mask_c, 0], X_s2[mask_c, 1],
                    c=[colors[c]], marker=markers[c % len(markers)],
                    s=55, edgecolors="white", linewidth=0.6,
                    label=f"{class_names[c]} (n={mask_c.sum()})", zorder=3)
    ax2.set_xlabel(STAGE2_FEATURES[0])
    ax2.set_ylabel(STAGE2_FEATURES[1])
    ax2.set_title("Stage 2: Alzheimer vs Glaucoma", fontsize=11)
    ax2.legend(fontsize=9, framealpha=0.9)

    fig.suptitle("Feature-Specific Hierarchical SVM — Decision Boundaries",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / f"decision_boundary.{FIGURE_FORMAT}"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Decision boundary saved to {out}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/5] Loading data ...")
    X_s1, X_s2, y, class_names, label_encoder = load_data()

    print(f"\n[2/5] Feature-Specific Hierarchical SVM — {N_REPEATS}×{N_SPLITS}-fold CV ...")
    print(f"  Stage 1: Normal vs Disease  → {STAGE1_FEATURES}")
    print(f"  Stage 2: Alzheimer vs Glaucoma → {STAGE2_FEATURES}")
    fold_metrics, agg_cm, roc_data, best_models = train_and_evaluate(
        X_s1, X_s2, y, class_names,
    )

    print("\n[3/5] Saving fold metrics ...")
    save_fold_csv(fold_metrics)

    print("\n[4/5] Generating plots ...")
    plot_confusion_matrix(agg_cm, class_names)
    plot_roc_curves(roc_data, class_names)
    plot_decision_boundary(X_s1, X_s2, y, best_models, class_names)

    print("\n[5/5] Saving model ...")
    joblib.dump(
        {
            "stage1_pipeline": best_models["stage1"],
            "stage2_pipeline": best_models["stage2"],
            "label_encoder": label_encoder,
            "stage1_features": STAGE1_FEATURES,
            "stage2_features": STAGE2_FEATURES,
            "class_names": class_names,
            "architecture": "feature_specific_hierarchical",
            "stage1_task": "Normal vs Disease (graph macro features)",
            "stage2_task": "Alzheimer vs Glaucoma (TDA features)",
        },
        MODEL_PATH,
    )
    print(f"  Model saved to {MODEL_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
