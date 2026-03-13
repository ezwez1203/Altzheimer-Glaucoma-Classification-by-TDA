"""
config.py — Constants and path configuration for graph extraction.
"""

from pathlib import Path

# project paths (relative to this file)
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
VCP_RESULTS  = PROJECT_ROOT / "01_VCP" / "results"

VESSEL_DIR  = str(VCP_RESULTS / "logs" / "vessel_png")
AV_DIR      = str(VCP_RESULTS / "logs" / "pixelwise_png")
OUTPUT_BASE = str(SCRIPT_DIR / "processed_graphs")

# AV mask color encoding (RGB)
COLOR_ARTERY = (255, 0, 0)   # red
COLOR_VEIN   = (0, 0, 255)   # blue

# AV label map values
LABEL_BG     = 0
LABEL_ARTERY = 1
LABEL_VEIN   = 2

# graph extraction defaults
MIN_EDGE_LENGTH = 3

# 8-connected neighbor offsets (dy, dx)
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]
