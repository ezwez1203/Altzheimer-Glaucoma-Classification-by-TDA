"""
config.py — Constants and path configuration for Track A (Radial Filtration).
"""

from pathlib import Path

# project paths
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

GRAPH_BASE = str(PROJECT_ROOT / "02_Graph_Extraction" / "processed_graphs")
OD_DIR     = str(PROJECT_ROOT / "01_VCP" / "results" / "logs" / "od_png")
OUTPUT_CSV = str(SCRIPT_DIR / "trackA_features.csv")

# persistence noise threshold — intervals shorter than this are discarded
NOISE_THRESHOLD = 5.0
