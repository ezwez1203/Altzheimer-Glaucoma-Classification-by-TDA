"""
main.py — Entry point for batch graph extraction.

Usage:
    conda activate cuda_tda
    python main.py
"""

import os

from config import VESSEL_DIR, AV_DIR, OUTPUT_BASE
from dataset import process_dataset


def main():
    # --- DRIVE ---
    drive_out = os.path.join(OUTPUT_BASE, "DRIVE")
    print("[1/2] Processing DRIVE dataset ...")
    process_dataset(
        vessel_dir=VESSEL_DIR,
        av_dir=AV_DIR,
        output_dir=drive_out,
        dataset_name="DRIVE",
    )

    # --- IOSTAR ---
    iostar_out = os.path.join(OUTPUT_BASE, "IOSTAR")
    print("[2/2] Processing IOSTAR dataset ...")
    process_dataset(
        vessel_dir=VESSEL_DIR,
        av_dir=AV_DIR,
        output_dir=iostar_out,
        dataset_name="IOSTAR",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
