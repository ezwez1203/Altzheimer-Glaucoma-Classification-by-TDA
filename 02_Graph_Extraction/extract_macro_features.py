"""
extract_macro_features.py — Extract macroscopic graph-level features.

Computes per-subject:
  - vessel_density      : vessel pixel ratio (from binary vessel mask)
  - avg_edge_thickness  : mean vessel thickness across all graph edges
  - total_junctions     : number of branching points (junction nodes)
  - total_endpoints     : number of terminal vessel endpoints
  - total_edges         : total vessel segment count
  - avg_edge_length     : mean vessel segment length (pixels)

These features capture the "forest-level" vascular architecture that
TDA features miss — useful for separating Normal from Disease (Stage 1).

Usage:
    cd 02_Graph_Extraction
    python extract_macro_features.py

Output:
    graph_macro_features.csv  (subject_id, dataset, 6 features)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ── Paths ──
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

GRAPH_DIR    = SCRIPT_DIR / "processed_graphs"
VESSEL_DIR   = PROJECT_ROOT / "01_VCP" / "results" / "logs" / "vessel_png"
OUTPUT_CSV   = SCRIPT_DIR / "graph_macro_features.csv"

DATASETS = ["DRIVE", "IOSTAR"]


def load_graph(pkl_path):
    """Load a NetworkX graph from pickle."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_vessel_density(subject_id, dataset):
    """Compute vessel pixel ratio from binary vessel mask."""
    vessel_path = VESSEL_DIR / f"{subject_id}_vessel.png"
    if not vessel_path.exists():
        print(f"    [WARN] Vessel mask not found: {vessel_path}")
        return np.nan

    img = np.array(Image.open(vessel_path).convert("L"))
    total_pixels = img.size
    vessel_pixels = np.count_nonzero(img > 127)
    return round(vessel_pixels / total_pixels, 6)


def extract_features_from_graph(G, subject_id, dataset):
    """Extract macroscopic features from a single NetworkX graph."""
    nodes = G.nodes(data=True)
    edges = G.edges(data=True)

    # Node-level counts
    junctions = sum(1 for _, d in nodes if d.get("node_type") == "junction")
    endpoints = sum(1 for _, d in nodes if d.get("node_type") == "endpoint")

    # Edge-level statistics
    edge_list = list(edges)
    n_edges = len(edge_list)

    if n_edges > 0:
        thicknesses = [d.get("avg_thickness", 0.0) for _, _, d in edge_list]
        lengths = [d.get("weight", 0) for _, _, d in edge_list]
        avg_thickness = round(float(np.mean(thicknesses)), 4)
        avg_length = round(float(np.mean(lengths)), 4)
    else:
        avg_thickness = 0.0
        avg_length = 0.0

    # Vessel density from the binary mask
    density = compute_vessel_density(subject_id, dataset)

    return {
        "subject_id": subject_id,
        "dataset": dataset,
        "vessel_density": density,
        "avg_edge_thickness": avg_thickness,
        "total_junctions": junctions,
        "total_endpoints": endpoints,
        "total_edges": n_edges,
        "avg_edge_length": avg_length,
    }


def main():
    rows = []

    for dataset in DATASETS:
        graph_dir = GRAPH_DIR / dataset
        if not graph_dir.exists():
            print(f"  [WARN] Graph directory not found: {graph_dir}")
            continue

        pkl_files = sorted(graph_dir.glob("*_graph.pkl"))
        print(f"  {dataset}: {len(pkl_files)} graphs")

        for pkl_path in pkl_files:
            subject_id = pkl_path.stem.replace("_graph", "")
            G = load_graph(pkl_path)
            feat = extract_features_from_graph(G, subject_id, dataset)
            rows.append(feat)
            print(f"    {subject_id:20s}  nodes={G.number_of_nodes():5d}  "
                  f"edges={G.number_of_edges():5d}  "
                  f"junctions={feat['total_junctions']:4d}  "
                  f"density={feat['vessel_density']:.4f}  "
                  f"thickness={feat['avg_edge_thickness']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
