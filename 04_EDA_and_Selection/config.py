"""
config.py — Paths and constants for EDA & Feature Selection.
"""

from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# inputs
TRACK_A_CSV    = str(PROJECT_ROOT / "03_trackA" / "trackA_features.csv")
TRACK_B_CSV    = str(PROJECT_ROOT / "03_trackB" / "trackB_features.csv")
GRAPH_MACRO_CSV = str(PROJECT_ROOT / "02_Graph_Extraction" / "graph_macro_features.csv")
LABELS_CSV     = str(SCRIPT_DIR / "labels.csv")

# outputs
PLOTS_DIR          = str(SCRIPT_DIR / "plots")
STATS_CSV          = str(SCRIPT_DIR / "statistical_results.csv")
SELECTED_CSV       = str(SCRIPT_DIR / "selected_features.csv")

# stage-specific outputs (hierarchical classification)
STATS_STAGE1_CSV   = str(SCRIPT_DIR / "statistical_results_stage1.csv")
STATS_STAGE2_CSV   = str(SCRIPT_DIR / "statistical_results_stage2.csv")
SELECTED_STAGE1_CSV = str(SCRIPT_DIR / "selected_features_stage1.csv")
SELECTED_STAGE2_CSV = str(SCRIPT_DIR / "selected_features_stage2.csv")

# statistical threshold
P_VALUE_THRESHOLD = 0.05

# feature columns (everything except identifiers)
ID_COLS = ["subject_id", "dataset"]

# plot settings
FIGURE_DPI   = 300
FIGURE_FORMAT = "png"
