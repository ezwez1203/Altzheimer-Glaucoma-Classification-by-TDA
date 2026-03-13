"""
main.py — Entry point for SVM classification on TDA features.

Usage:
    conda activate cuda_tda
    python main.py
"""

import os
import joblib

from config import RESULTS_DIR, MODEL_PATH
from classifier import TopologicalSVMClassifier
from plots import plot_confusion_matrix, plot_roc_curves


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    clf = TopologicalSVMClassifier()

    # step 1 — load
    print("[1/4] Loading data ...")
    clf.load_and_preprocess()

    # step 2+3 — cross-validated grid search + evaluation
    print("\n[2/4] Training with stratified K-fold + GridSearchCV ...")
    metrics = clf.train_and_evaluate()

    # step 3 — plots
    print("\n[3/4] Generating plots ...")
    plot_confusion_matrix(clf.aggregate_cm, clf.class_names)
    plot_roc_curves(clf.roc_data, clf.class_names)

    # step 4 — save model
    print("\n[4/4] Saving model ...")
    joblib.dump(
        {
            "pipeline": clf.best_pipeline,
            "label_encoder": clf.label_encoder,
            "feature_names": clf.feature_names,
            "class_names": clf.class_names,
        },
        MODEL_PATH,
    )
    print(f"  Model saved to {MODEL_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
