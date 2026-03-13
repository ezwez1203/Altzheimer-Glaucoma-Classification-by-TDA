"""
dataset.py — File matching, batch processing, and I/O for graph extraction.
"""

import os
import pickle
import re
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from extractor import RetinalGraphExtractor


def find_subject_pairs(
    vessel_dir: str,
    av_dir: str,
    dataset_filter: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Match vessel masks to AV masks by subject ID.

    Naming convention:
        vessel: {subject_id}_vessel.png
        av:     {subject_id}_av_pixelwise.png

    Args:
        vessel_dir:     directory with *_vessel.png files.
        av_dir:         directory with *_av_pixelwise.png files.
        dataset_filter: "DRIVE" keeps only XX_test IDs,
                        "IOSTAR" keeps only STAR* IDs,
                        None keeps all.

    Returns:
        List of dicts with keys "id", "vessel", "av".
    """
    vessel_files = {f for f in os.listdir(vessel_dir) if f.endswith("_vessel.png")}
    av_files     = {f for f in os.listdir(av_dir) if f.endswith("_av_pixelwise.png")}

    pairs = []
    for vf in sorted(vessel_files):
        subject_id = vf.replace("_vessel.png", "")

        if dataset_filter == "DRIVE" and not re.match(r"^\d{2}_test$", subject_id):
            continue
        if dataset_filter == "IOSTAR" and not subject_id.startswith("STAR"):
            continue

        av_name = f"{subject_id}_av_pixelwise.png"
        if av_name in av_files:
            pairs.append({
                "id": subject_id,
                "vessel": os.path.join(vessel_dir, vf),
                "av": os.path.join(av_dir, av_name),
            })
        else:
            print(f"  [WARN] no AV mask for {subject_id}, skipping")

    return pairs


def process_dataset(
    vessel_dir: str,
    av_dir: str,
    output_dir: str,
    dataset_name: str = "",
    min_edge_length: int = 3,
) -> None:
    """
    Batch-process all subjects for one dataset.

    Args:
        vessel_dir:      directory containing *_vessel.png files.
        av_dir:          directory containing *_av_pixelwise.png files.
        output_dir:      where to save .pkl graph files.
        dataset_name:    "DRIVE" or "IOSTAR" (used for filtering + logging).
        min_edge_length: passed to RetinalGraphExtractor.
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = RetinalGraphExtractor(min_edge_length=min_edge_length)
    pairs = find_subject_pairs(vessel_dir, av_dir, dataset_filter=dataset_name or None)

    if not pairs:
        print(f"  [{dataset_name}] no matching pairs found.")
        return

    print(f"  [{dataset_name}] found {len(pairs)} subjects")

    for p in pairs:
        sid = p["id"]
        vessel_mask = np.array(Image.open(p["vessel"]).convert("L"))
        av_mask     = np.array(Image.open(p["av"]).convert("RGB"))

        graph = extractor.extract_graph(vessel_mask, av_mask)

        out_path = os.path.join(output_dir, f"{sid}_graph.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        print(f"    {sid}: {n_nodes} nodes, {n_edges} edges -> {out_path}")
