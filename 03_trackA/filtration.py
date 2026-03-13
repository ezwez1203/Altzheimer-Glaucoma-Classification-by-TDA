"""
filtration.py — Radial filtration value assignment and simplex tree construction.
"""

import math
from typing import Tuple

import gudhi
import networkx as nx
import numpy as np


def compute_od_center(od_mask: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the center of mass of the optic disc mask.

    Args:
        od_mask: (H, W) binary mask (0/255 or 0/1).

    Returns:
        (y_c, x_c) center coordinates. Falls back to image center
        if the mask is empty or None.
    """
    if od_mask is None:
        return None

    binary = (od_mask > 0).astype(np.uint8)
    ys, xs = np.where(binary == 1)

    if len(ys) == 0:
        return None

    return float(np.mean(ys)), float(np.mean(xs))


def assign_filtration_values(
    graph: nx.Graph,
    od_center: Tuple[float, float],
) -> nx.Graph:
    """
    Assign radial filtration values to every node and edge.

    Node filtration = Euclidean distance from OD center.
    Edge filtration = max(f(u), f(v)) to satisfy the simplex property.

    Args:
        graph:     input networkx graph (nodes have 'y', 'x' attributes).
        od_center: (y_c, x_c) optic disc center.

    Returns:
        The same graph with 'filtration_val' set on nodes and edges.
    """
    yc, xc = od_center

    for node, data in graph.nodes(data=True):
        ny = float(data["y"])
        nx_ = float(data["x"])
        dist = math.sqrt((ny - yc) ** 2 + (nx_ - xc) ** 2)
        graph.nodes[node]["filtration_val"] = dist

    for u, v in graph.edges():
        fu = graph.nodes[u]["filtration_val"]
        fv = graph.nodes[v]["filtration_val"]
        graph.edges[u, v]["filtration_val"] = max(fu, fv)

    return graph


def build_simplex_tree(graph: nx.Graph) -> gudhi.SimplexTree:
    """
    Construct a GUDHI SimplexTree from a filtration-annotated graph.

    Args:
        graph: networkx graph with 'filtration_val' on all nodes and edges.

    Returns:
        Populated gudhi.SimplexTree ready for persistence computation.
    """
    st = gudhi.SimplexTree()

    # node → vertex id mapping (nodes are (y,x) tuples, simplex tree needs ints)
    node_to_id = {}
    for i, (node, data) in enumerate(graph.nodes(data=True)):
        node_to_id[node] = i
        st.insert([i], filtration=data["filtration_val"])

    for u, v, data in graph.edges(data=True):
        uid = node_to_id[u]
        vid = node_to_id[v]
        st.insert([uid, vid], filtration=data["filtration_val"])

    return st
