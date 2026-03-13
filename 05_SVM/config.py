"""
config.py — Paths and constants for the SVM classifier.
"""

from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# input
SELECTED_CSV = str(PROJECT_ROOT / "04_EDA_and_Selection" / "selected_features.csv")

# outputs
RESULTS_DIR  = str(SCRIPT_DIR / "results")
MODEL_PATH   = str(SCRIPT_DIR / "results" / "best_svm_model.pkl")
METRICS_CSV  = str(SCRIPT_DIR / "results" / "fold_metrics.csv")

# identifiers (not features)
ID_COLS   = ["subject_id", "dataset"]
LABEL_COL = "label"

# cross-validation
N_SPLITS    = 5
RANDOM_SEED = 42

# grid search parameter space
# keys are prefixed with 'svc__' because the SVC lives inside a Pipeline
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

# plot settings
FIGURE_DPI    = 300
FIGURE_FORMAT = "png"
