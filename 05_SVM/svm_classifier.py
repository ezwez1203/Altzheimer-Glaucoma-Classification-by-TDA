"""
svm_classifier.py — Robust SVM classification pipeline for retinal TDA features.

Standalone script that performs:
  1. Data loading from selected_features.csv
  2. Pipeline-based StandardScaler → SVC (scaling inside CV = no data leakage)
  3. Repeated Stratified K-Fold CV with nested GridSearchCV
  4. Publication-ready plots: confusion matrix, ROC curves, decision boundary
  5. Model export and per-fold metrics CSV

Design decisions for small medical datasets (N ≈ 50, 3-class):
  - sklearn.pipeline.Pipeline wraps scaler + SVC so fit_transform() only
    touches the training fold → prevents data leakage.
  - class_weight='balanced' compensates for imbalance (Normal 23 / AD 20 /
    Glaucoma 7) at the loss level. SMOTE is avoided because synthetic
    oversampling on N=50 amplifies noise and worsens overfitting.
  - RepeatedStratifiedKFold (3 repeats × 5 folds = 15 evaluations) stabilises
    variance compared to a single 5-fold split.
  - probability=True enables Platt-scaled OVR AUROC computation.
  - With exactly 2 selected features we also produce a decision boundary plot.

Usage:
    cd 05_SVM
    python svm_classifier.py
"""

import os
import sys
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
    classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SELECTED_CSV = PROJECT_ROOT / "04_EDA_and_Selection" / "selected_features.csv"
RESULTS_DIR  = SCRIPT_DIR / "results"
MODEL_PATH   = RESULTS_DIR / "best_svm_model.pkl"
METRICS_CSV  = RESULTS_DIR / "fold_metrics.csv"

ID_COLS   = ["subject_id", "dataset"]
LABEL_COL = "label"

# Repeated Stratified K-Fold: 3 repeats × 5 folds = 15 evaluations
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


# ──────────────────────────────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────────────────────────────

def load_data(csv_path=SELECTED_CSV):
    """Load selected features CSV; return X, y (encoded), feature names,
    class names, and fitted LabelEncoder."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples from {csv_path}")

    drop = [c for c in ID_COLS + [LABEL_COL] if c in df.columns]
    feature_names = [c for c in df.columns if c not in drop]

    X = df[feature_names].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].values)
    class_names = list(le.classes_)

    counts = dict(zip(class_names, [int(c) for c in np.bincount(y)]))
    print(f"  Features    : {feature_names}")
    print(f"  Classes     : {class_names}")
    print(f"  Distribution: {counts}")

    return X, y, feature_names, class_names, le


# ──────────────────────────────────────────────────────────────────────
#  Core Training — Repeated Stratified K-Fold + Nested GridSearchCV
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


def train_and_evaluate(X, y, class_names):
    """
    Repeated Stratified K-Fold cross-validation with nested GridSearchCV.

    Outer loop: RepeatedStratifiedKFold (3 × 5 = 15 evaluations)
        → unbiased performance estimation with reduced variance
    Inner loop: GridSearchCV on each training split
        → hyperparameter tuning

    Pipeline(StandardScaler → SVC) ensures scaler is re-fitted on
    the training fold only — no data leakage.

    Returns:
        fold_metrics, aggregate_cm, roc_data, best_pipeline
    """
    n_classes = len(class_names)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED,
    )

    fold_metrics = []
    # For ROC / CM we use a single representative pass (first repeat)
    first_pass_y_true, first_pass_y_pred = [], []
    first_pass_y_prob = np.zeros((len(y), n_classes))
    first_pass_counts = np.zeros(len(y), dtype=int)

    total_folds = N_SPLITS * N_REPEATS
    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Inner CV folds capped by smallest class in training split
        min_class = int(np.bincount(y_tr).min())
        inner_k = max(2, min(3, min_class))

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ])

        gs = GridSearchCV(
            pipe, PARAM_GRID,
            cv=StratifiedKFold(inner_k, shuffle=True,
                               random_state=RANDOM_SEED),
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_tr, y_tr)

        y_pred = gs.predict(X_te)
        y_prob = gs.predict_proba(X_te)

        # Accumulate probabilities for ROC (average across repeats)
        first_pass_y_prob[test_idx] += y_prob
        first_pass_counts[test_idx] += 1

        # Only track confusion matrix from the first repeat
        if fold_i < N_SPLITS:
            first_pass_y_true.extend(y_te)
            first_pass_y_pred.extend(y_pred)

        acc  = accuracy_score(y_te, y_pred)
        f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)
        sens = recall_score(y_te, y_pred, average="macro", zero_division=0)
        spec = _macro_specificity(y_te, y_pred, n_classes)
        prec = precision_score(y_te, y_pred, average="macro", zero_division=0)

        best_p = {k.replace("svc__", ""): v
                  for k, v in gs.best_params_.items()}

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
            "best_params": best_p,
        })
        print(f"  Repeat {repeat_num} Fold {fold_num}/{N_SPLITS}  "
              f"Acc={acc:.3f}  F1={f1:.3f}  Sens={sens:.3f}  "
              f"Spec={spec:.3f}  params={best_p}")

    # ── Aggregate confusion matrix (first repeat only) ──
    aggregate_cm = confusion_matrix(
        np.array(first_pass_y_true), np.array(first_pass_y_pred),
        labels=range(n_classes),
    )

    # ── ROC data (averaged probabilities across repeats) ──
    mask = first_pass_counts > 0
    first_pass_y_prob[mask] /= first_pass_counts[mask, np.newaxis]
    roc_data = _compute_roc(y, first_pass_y_prob, n_classes)

    # ── Final model on full dataset ──
    min_class_all = int(np.bincount(y).min())
    inner_k_all = max(2, min(N_SPLITS, min_class_all))

    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])
    final_gs = GridSearchCV(
        final_pipe, PARAM_GRID,
        cv=StratifiedKFold(inner_k_all, shuffle=True,
                           random_state=RANDOM_SEED),
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    final_gs.fit(X, y)
    best_pipeline = final_gs.best_estimator_

    # ── Summary ──
    metric_keys = ["accuracy", "sensitivity", "specificity", "precision", "f1_macro"]
    print(f"\n  === {N_REPEATS}×{N_SPLITS}-Fold Cross-Validation Results ===")
    for k in metric_keys:
        vals = [m[k] for m in fold_metrics]
        print(f"    {k:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    macro_auc = roc_data.get("macro_auc", 0.0)
    print(f"    {'auroc_macro':20s}: {macro_auc:.4f}")

    best_final = {k.replace("svc__", ""): v
                  for k, v in final_gs.best_params_.items()}
    print(f"  Final model params: {best_final}")

    return fold_metrics, aggregate_cm, roc_data, best_pipeline


# ──────────────────────────────────────────────────────────────────────
#  Metrics CSV Export
# ──────────────────────────────────────────────────────────────────────

def save_fold_csv(fold_metrics):
    """Export per-fold metrics + aggregate (mean ± std) to CSV."""
    rows = []
    for m in fold_metrics:
        rows.append({
            "repeat": m["repeat"],
            "fold": m["fold"],
            "accuracy": round(m["accuracy"], 4),
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "precision": round(m["precision"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "kernel": m["best_params"].get("kernel", ""),
            "C": m["best_params"].get("C", ""),
            "gamma": m["best_params"].get("gamma", ""),
        })

    metric_keys = ["accuracy", "sensitivity", "specificity", "precision", "f1_macro"]
    agg_mean = {k: round(np.mean([m[k] for m in fold_metrics]), 4)
                for k in metric_keys}
    agg_std = {k: round(np.std([m[k] for m in fold_metrics]), 4)
               for k in metric_keys}

    rows.append({
        "repeat": "mean", "fold": "",
        **agg_mean, "kernel": "", "C": "", "gamma": "",
    })
    rows.append({
        "repeat": "std", "fold": "",
        **agg_std, "kernel": "", "C": "", "gamma": "",
    })

    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
    print(f"  Fold metrics saved to {METRICS_CSV}")


# ──────────────────────────────────────────────────────────────────────
#  Visualisation — Publication-Quality Plots
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
    ax.set_title("Confusion Matrix (K-Fold Aggregated)", fontsize=12, pad=10)

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


def plot_decision_boundary(X, y, pipeline, class_names, feature_names):
    """
    2D decision boundary plot — only possible when n_features == 2.

    Shows the SVM decision regions overlaid with scatter points,
    coloured by true class. Uses the full trained pipeline (scaler + SVC).
    """
    if X.shape[1] != 2:
        print("  Skipping decision boundary (requires exactly 2 features)")
        return

    _setup_style()

    # Create a mesh in the original (unscaled) feature space
    margin = 0.1
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    xx, yy = np.meshgrid(
        np.linspace(x_min - margin * x_range, x_max + margin * x_range, 300),
        np.linspace(y_min - margin * y_range, y_max + margin * y_range, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict on grid (pipeline handles scaling internally)
    Z = pipeline.predict(grid).reshape(xx.shape)

    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    cmap_bg = matplotlib.colors.ListedColormap(
        [c + "33" for c in colors[:len(class_names)]]
    )
    cmap_pt = matplotlib.colors.ListedColormap(colors[:len(class_names)])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap_bg, levels=np.arange(
        len(class_names) + 1) - 0.5)
    ax.contour(xx, yy, Z, colors="gray", linewidths=0.5, alpha=0.5)

    markers = ["o", "s", "D"]
    for c in range(len(class_names)):
        mask = y == c
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=[colors[c]], marker=markers[c % len(markers)],
            s=55, edgecolors="white", linewidth=0.6,
            label=f"{class_names[c]} (n={mask.sum()})", zorder=3,
        )

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("SVM Decision Boundary", fontsize=12, pad=10)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

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

    # Step 1 — Load
    print("[1/5] Loading data ...")
    X, y, feature_names, class_names, label_encoder = load_data()

    # Step 2 — Train + evaluate (repeated stratified K-fold)
    print(f"\n[2/5] Training with {N_REPEATS}×{N_SPLITS}-fold "
          "stratified CV + GridSearchCV ...")
    fold_metrics, agg_cm, roc_data, best_pipeline = train_and_evaluate(
        X, y, class_names,
    )

    # Step 3 — Metrics CSV
    print("\n[3/5] Saving fold metrics ...")
    save_fold_csv(fold_metrics)

    # Step 4 — Plots
    print("\n[4/5] Generating plots ...")
    plot_confusion_matrix(agg_cm, class_names)
    plot_roc_curves(roc_data, class_names)
    plot_decision_boundary(X, y, best_pipeline, class_names, feature_names)

    # Step 5 — Save model
    print("\n[5/5] Saving model ...")
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "class_names": class_names,
        },
        MODEL_PATH,
    )
    print(f"  Model saved to {MODEL_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
