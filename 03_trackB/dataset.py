"""
dataset.py — File matching, batch processing, and CSV export for Track B.
"""

import os
import pickle
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from extractor import AlzheimerTDAExtractor


def find_subject_files(
    graph_dir: str,
    vessel_dir: str,
    av_dir: str,
    dataset_filter: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Match graph, vessel mask, and AV mask files by subject ID.

    Naming conventions:
        graph:  {id}_graph.pkl
        vessel: {id}_vessel.png
        av:     {id}_av_pixelwise.png
    """
    graph_files  = {f for f in os.listdir(graph_dir) if f.endswith("_graph.pkl")}
    vessel_files = {f for f in os.listdir(vessel_dir) if f.endswith("_vessel.png")}
    av_files     = {f for f in os.listdir(av_dir) if f.endswith("_av_pixelwise.png")}

    subjects = []
    for gf in sorted(graph_files):
        sid = gf.replace("_graph.pkl", "")

        if dataset_filter == "DRIVE" and not re.match(r"^\d{2}_test$", sid):
            continue
        if dataset_filter == "IOSTAR" and not sid.startswith("STAR"):
            continue

        vessel_name = f"{sid}_vessel.png"
        av_name = f"{sid}_av_pixelwise.png"

        if vessel_name not in vessel_files:
            print(f"  [WARN] no vessel mask for {sid}, skipping")
            continue

        subjects.append({
            "id": sid,
            "graph": os.path.join(graph_dir, gf),
            "vessel": os.path.join(vessel_dir, vessel_name),
            "av": os.path.join(av_dir, av_name) if av_name in av_files else None,
        })

    return subjects


def process_dataset(
    graph_dir: str,
    vessel_dir: str,
    av_dir: str,
    dataset_name: str = "",
    noise_threshold: float = 5.0,
) -> List[Dict]:
    """
    Batch-process all subjects for one dataset.

    Args:
        graph_dir:       directory with *_graph.pkl files.
        vessel_dir:      directory with *_vessel.png files.
        av_dir:          directory with *_av_pixelwise.png files.
        dataset_name:    "DRIVE" or "IOSTAR".
        noise_threshold: passed to flooding TDA.

    Returns:
        List of feature dicts.
    """
    extractor = AlzheimerTDAExtractor(noise_threshold=noise_threshold)
    subjects = find_subject_files(graph_dir, vessel_dir, av_dir, dataset_filter=dataset_name or None)

    if not subjects:
        print(f"  [{dataset_name}] no subjects found.")
        return []

    print(f"  [{dataset_name}] found {len(subjects)} subjects")
    rows = []

    for s in subjects:
        sid = s["id"]

        with open(s["graph"], "rb") as f:
            graph = pickle.load(f)

        vessel_mask = np.array(Image.open(s["vessel"]).convert("L"))

        av_mask = None
        if s["av"] is not None:
            av_mask = np.array(Image.open(s["av"]).convert("RGB"))

        features = extractor.extract_features(graph, vessel_mask, av_mask)
        features["subject_id"] = sid
        features["dataset"] = dataset_name
        rows.append(features)

        print(f"    {sid}: D={features['fractal_dimension']:.4f}  "
              f"AVR={features['AVR']:.4f}  "
              f"flood_b0_max={features['flood_b0_max_lifespan']:.2f}")

    return rows


def export_csv(rows: List[Dict], output_path: str) -> None:
    """Save feature rows to CSV."""
    df = pd.DataFrame(rows)
    col_order = [
        "subject_id", "dataset",
        "fractal_dimension", "AVR",
        "flood_b0_max_lifespan", "flood_b0_sum_lifespan",
        "flood_b1_count", "flood_b1_max_lifespan", "flood_persistence_entropy",
    ]
    df = df[col_order]
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows -> {output_path}")
