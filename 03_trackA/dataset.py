"""
dataset.py — File matching, batch processing, and CSV export for Track A.
"""

import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from extractor import RadialFiltrationTDA


def find_subject_files(
    graph_dir: str,
    od_dir: str,
) -> List[Dict[str, str]]:
    """
    Match graph .pkl files to OD mask .png files by subject ID.

    Naming convention:
        graph:   {subject_id}_graph.pkl
        od mask: {subject_id}_od.png

    Returns:
        List of dicts with keys "id", "graph", "od".
    """
    graph_files = {f for f in os.listdir(graph_dir) if f.endswith("_graph.pkl")}
    od_files    = {f for f in os.listdir(od_dir) if f.endswith("_od.png")}

    subjects = []
    for gf in sorted(graph_files):
        sid = gf.replace("_graph.pkl", "")
        od_name = f"{sid}_od.png"

        entry = {
            "id": sid,
            "graph": os.path.join(graph_dir, gf),
            "od": os.path.join(od_dir, od_name) if od_name in od_files else None,
        }
        subjects.append(entry)

    return subjects


def process_dataset(
    graph_dir: str,
    od_dir: str,
    dataset_name: str = "",
    noise_threshold: float = 5.0,
) -> List[Dict]:
    """
    Batch-process all subjects for one dataset.

    Args:
        graph_dir:       directory with *_graph.pkl files.
        od_dir:          directory with *_od.png files.
        dataset_name:    label for logging and the 'dataset' column.
        noise_threshold: passed to RadialFiltrationTDA.

    Returns:
        List of feature dicts (one per subject).
    """
    tda = RadialFiltrationTDA(noise_threshold=noise_threshold)
    subjects = find_subject_files(graph_dir, od_dir)

    if not subjects:
        print(f"  [{dataset_name}] no subjects found.")
        return []

    print(f"  [{dataset_name}] found {len(subjects)} subjects")
    rows = []

    for s in subjects:
        sid = s["id"]

        # load graph
        with open(s["graph"], "rb") as f:
            graph = pickle.load(f)

        # load OD mask (or None)
        od_mask = None
        if s["od"] is not None and os.path.exists(s["od"]):
            od_mask = np.array(Image.open(s["od"]).convert("L"))

        # infer image shape from OD mask or first node
        if od_mask is not None:
            img_shape = od_mask.shape
        else:
            first = list(graph.nodes(data=True))[0][1] if graph.number_of_nodes() > 0 else {}
            img_shape = (int(first.get("y", 292)) * 2, int(first.get("x", 282)) * 2)

        features = tda.extract_features(graph, od_mask, image_shape=img_shape)
        features["subject_id"] = sid
        features["dataset"] = dataset_name
        rows.append(features)

        print(f"    {sid}: b0_max={features['b0_max_lifespan']:.1f}  "
              f"b1_count={features['b1_count']}  "
              f"entropy={features['persistence_entropy']:.4f}")

    return rows


def export_csv(rows: List[Dict], output_path: str) -> None:
    """Save feature rows to CSV."""
    df = pd.DataFrame(rows)
    col_order = [
        "subject_id", "dataset",
        "b0_max_lifespan", "b0_sum_lifespan",
        "b1_count", "b1_max_lifespan", "persistence_entropy",
    ]
    df = df[col_order]
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows -> {output_path}")
