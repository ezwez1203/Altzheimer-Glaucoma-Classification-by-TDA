"""
graph_builder.py — Assemble a networkx graph from traced edges + attribute maps.
"""

from typing import List, Set, Tuple

import networkx as nx
import numpy as np

from config import LABEL_ARTERY, LABEL_VEIN


def build_av_label_map(av_mask: np.ndarray) -> np.ndarray:
    """
    Convert an RGB AV mask to an integer label map.
        0 = background, 1 = artery (red), 2 = vein (blue)

    Args:
        av_mask: (H, W, 3) uint8 RGB image.

    Returns:
        (H, W) uint8 label array.
    """
    label = np.zeros(av_mask.shape[:2], dtype=np.uint8)
    label[(av_mask[:, :, 0] == 255) & (av_mask[:, :, 2] == 0)] = LABEL_ARTERY
    label[(av_mask[:, :, 2] == 255) & (av_mask[:, :, 0] == 0)] = LABEL_VEIN
    return label


def majority_vessel_type(av_label: np.ndarray, ys: list, xs: list) -> str:
    """
    Determine artery / vein by majority vote along edge pixels.

    Args:
        av_label: (H, W) integer label map.
        ys, xs:   row and column indices of the edge pixels.

    Returns:
        "artery", "vein", or "unknown".
    """
    labels = av_label[ys, xs]
    artery_count = int(np.sum(labels == LABEL_ARTERY))
    vein_count   = int(np.sum(labels == LABEL_VEIN))
    if artery_count == 0 and vein_count == 0:
        return "unknown"
    return "artery" if artery_count >= vein_count else "vein"


def build_graph(
    edges: List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]],
    endpoints: Set[Tuple[int, int]],
    junctions: Set[Tuple[int, int]],
    thickness_map: np.ndarray,
    av_label: np.ndarray,
) -> nx.Graph:
    """
    Construct a networkx.Graph from traced edges with morphometric attributes.

    Node attributes:
        y, x       — pixel coordinates
        node_type  — "endpoint" or "junction"

    Edge attributes:
        weight         — edge length in pixels
        avg_thickness  — mean vessel thickness along the edge
        vessel_type    — "artery", "vein", or "unknown"
        path           — list of (y, x) pixel coordinates

    Args:
        edges:          list of (start, end, pixel_path).
        endpoints:      set of endpoint (y, x) coordinates.
        junctions:      set of junction (y, x) coordinates.
        thickness_map:  (H, W) float thickness array.
        av_label:       (H, W) integer AV label array.

    Returns:
        Populated networkx.Graph.
    """
    G = nx.Graph()

    all_nodes = endpoints | junctions
    for node in all_nodes:
        ntype = "endpoint" if node in endpoints else "junction"
        G.add_node(node, y=node[0], x=node[1], node_type=ntype)

    for start, end, path in edges:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]

        length = len(path)
        avg_thick = float(np.mean(thickness_map[ys, xs]))
        vtype = majority_vessel_type(av_label, ys, xs)

        G.add_edge(
            start,
            end,
            weight=length,
            avg_thickness=round(avg_thick, 3),
            vessel_type=vtype,
            path=path,
        )

    return G
