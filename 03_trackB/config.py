"""
config.py — Constants and path configuration for Track B (Alzheimer screening).
"""

from pathlib import Path

# project paths
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

GRAPH_BASE  = str(PROJECT_ROOT / "02_Graph_Extraction" / "processed_graphs")
VESSEL_DIR  = str(PROJECT_ROOT / "01_VCP" / "results" / "logs" / "vessel_png")
AV_DIR      = str(PROJECT_ROOT / "01_VCP" / "results" / "logs" / "pixelwise_png")
OUTPUT_CSV  = str(SCRIPT_DIR / "trackB_features.csv")

# box-counting: range of box sizes (powers of 2)
BOX_SIZES = [2, 4, 8, 16, 32, 64, 128]

# flooding persistence noise threshold
NOISE_THRESHOLD = 5.0

# AV label encoding
LABEL_ARTERY = 1
LABEL_VEIN   = 2
