"""
main.py — Entry point for Track B: Alzheimer's Disease biomarker extraction.

Usage:
    conda activate cuda_tda
    python main.py
"""

import os

from config import GRAPH_BASE, VESSEL_DIR, AV_DIR, OUTPUT_CSV
from dataset import process_dataset, export_csv


def main():
    all_rows = []

    # --- DRIVE ---
    drive_graphs = os.path.join(GRAPH_BASE, "DRIVE")
    print("[1/2] Processing DRIVE ...")
    rows = process_dataset(
        graph_dir=drive_graphs,
        vessel_dir=VESSEL_DIR,
        av_dir=AV_DIR,
        dataset_name="DRIVE",
    )
    all_rows.extend(rows)

    # --- IOSTAR ---
    iostar_graphs = os.path.join(GRAPH_BASE, "IOSTAR")
    print("[2/2] Processing IOSTAR ...")
    rows = process_dataset(
        graph_dir=iostar_graphs,
        vessel_dir=VESSEL_DIR,
        av_dir=AV_DIR,
        dataset_name="IOSTAR",
    )
    all_rows.extend(rows)

    # export
    export_csv(all_rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()
