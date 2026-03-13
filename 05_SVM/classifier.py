"""
classifier.py — Robust SVM classification with Pipeline-based scaling,
                stratified K-fold CV, and nested GridSearchCV.

Design decisions for small medical datasets (N ≈ 50):
  - sklearn.pipeline.Pipeline wraps StandardScaler + SVC so that
    scaling is always fitted on the training fold only → no data leakage.
  - class_weight='balanced' compensates for imbalanced class sizes
    (Normal 23 / Alzheimer 20 / Glaucoma 7) without synthetic oversampling.
  - Inner CV folds are capped at the smallest class count in the
    training split to prevent StratifiedKFold from failing.
  - probability=True enables Platt scaling for OVR AUROC computation.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)

from config import (
    SELECTED_CSV, ID_COLS, LABEL_COL,
    N_SPLITS, RANDOM_SEED, PARAM_GRID, METRICS_CSV,
)


class TopologicalSVMClassifier:
    """
    SVM classifier for retinal TDA features.

    Uses sklearn Pipeline(StandardScaler → SVC) inside stratified K-fold
    cross-validation with GridSearchCV hyperparameter tuning.
    Designed for small, imbalanced medical datasets (~50 subjects).
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.feature_names = None
        self.class_names = None
        self.label_encoder = LabelEncoder()

        # populated after training
        self.best_pipeline = None
        self.fold_metrics = []
        self.aggregate_cm = None
        self.roc_data = None

    # ------------------------------------------------------------------ #
    #  Step 1 — Data loading                                              #
    # ------------------------------------------------------------------ #

    def load_and_preprocess(self, csv_path: str = SELECTED_CSV) -> None:
        """
        Load the selected-features CSV and separate X / y.

        Encoding and scaling happen inside the Pipeline during training,
        so this method only parses the DataFrame.
        """
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} samples from {csv_path}")

        drop_cols = [c for c in ID_COLS + [LABEL_COL] if c in df.columns]
        self.feature_names = [c for c in df.columns if c not in drop_cols]
        self.X = df[self.feature_names].values.astype(np.float64)
        self.y = self.label_encoder.fit_transform(df[LABEL_COL].values)
        self.class_names = list(self.label_encoder.classes_)

        counts = dict(zip(self.class_names,
                          [int(c) for c in np.bincount(self.y)]))
        print(f"  Features    : {self.feature_names}")
        print(f"  Classes     : {self.class_names}")
        print(f"  Distribution: {counts}")

    # ------------------------------------------------------------------ #
    #  Step 2 — Cross-validated grid search                               #
    # ------------------------------------------------------------------ #

    def train_and_evaluate(self) -> Dict[str, float]:
        """
        Nested cross-validation:
          outer loop  — StratifiedKFold(5) for unbiased evaluation
          inner loop  — GridSearchCV on train split for hyperparameter tuning

        The Pipeline(StandardScaler → SVC) ensures scaling is always
        re-fitted on the training fold alone.

        After CV, a final Pipeline is trained on the full dataset
        with the globally best hyperparameters (for model export).
        """
        outer_cv = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED,
        )
        n_classes = len(self.class_names)

        self.fold_metrics = []
        all_y_true, all_y_pred = [], []
        all_y_prob = np.zeros((len(self.y), n_classes))

        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(self.X, self.y)):
            X_tr, X_te = self.X[train_idx], self.X[test_idx]
            y_tr, y_te = self.y[train_idx], self.y[test_idx]

            # inner CV folds limited by smallest class in training split
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

            all_y_true.extend(y_te)
            all_y_pred.extend(y_pred)
            all_y_prob[test_idx] = y_prob

            acc  = accuracy_score(y_te, y_pred)
            f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)
            sens = recall_score(y_te, y_pred, average="macro", zero_division=0)
            spec = self._macro_specificity(y_te, y_pred, n_classes)

            best_p = {k.replace("svc__", ""): v
                      for k, v in gs.best_params_.items()}

            self.fold_metrics.append({
                "fold": fold_i + 1,
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "f1_macro": f1,
                "best_params": best_p,
            })
            print(f"  Fold {fold_i + 1}/{N_SPLITS}  "
                  f"Acc={acc:.3f}  F1={f1:.3f}  Sens={sens:.3f}  "
                  f"Spec={spec:.3f}  params={best_p}")

        # ---- aggregate results ---- #
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        self.aggregate_cm = confusion_matrix(all_y_true, all_y_pred,
                                             labels=range(n_classes))
        self.roc_data = self._compute_roc(all_y_true, all_y_prob, n_classes)

        # ---- final model on full data ---- #
        min_class_all = int(np.bincount(self.y).min())
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
        final_gs.fit(self.X, self.y)
        self.best_pipeline = final_gs.best_estimator_

        # ---- print summary ---- #
        metrics = self._aggregate_metrics()
        print(f"\n  === {N_SPLITS}-Fold Cross-Validation Results ===")
        for k, v in metrics.items():
            print(f"    {k:20s}: {v:.4f}")

        macro_auc = self.roc_data.get("macro_auc", 0.0)
        print(f"    {'auroc_macro':20s}: {macro_auc:.4f}")

        best_final = {k.replace("svc__", ""): v
                      for k, v in final_gs.best_params_.items()}
        print(f"  Final model params: {best_final}")

        # save fold-level CSV
        self._save_fold_csv()

        return metrics

    # ------------------------------------------------------------------ #
    #  helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
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

    @staticmethod
    def _compute_roc(y_true, y_prob, n_classes):
        """Per-class ROC curves + macro AUROC (one-vs-rest)."""
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
            result["macro_auc"] = float(np.mean(
                [result[c]["auc"] for c in range(n_classes)]))

        return result

    def _aggregate_metrics(self) -> Dict[str, float]:
        keys = ["accuracy", "sensitivity", "specificity", "f1_macro"]
        return {
            k: float(np.mean([m[k] for m in self.fold_metrics]))
            for k in keys
        }

    def _save_fold_csv(self) -> None:
        """Export per-fold metrics to CSV for paper tables."""
        rows = []
        for m in self.fold_metrics:
            rows.append({
                "fold": m["fold"],
                "accuracy": round(m["accuracy"], 4),
                "sensitivity": round(m["sensitivity"], 4),
                "specificity": round(m["specificity"], 4),
                "f1_macro": round(m["f1_macro"], 4),
                "kernel": m["best_params"].get("kernel", ""),
                "C": m["best_params"].get("C", ""),
                "gamma": m["best_params"].get("gamma", ""),
            })

        # append aggregate row
        agg = self._aggregate_metrics()
        rows.append({
            "fold": "mean",
            "accuracy": round(agg["accuracy"], 4),
            "sensitivity": round(agg["sensitivity"], 4),
            "specificity": round(agg["specificity"], 4),
            "f1_macro": round(agg["f1_macro"], 4),
            "kernel": "",
            "C": "",
            "gamma": "",
        })

        pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
        print(f"  Fold metrics saved to {METRICS_CSV}")
