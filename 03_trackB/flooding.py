"""
flooding.py — Flooding filtration and persistent homology feature extraction.

Simulates a flooding process where thicker vessels fill first.
Node filtration = max avg_thickness of incident edges.
Edge filtration = min(f(u), f(v)) for a valid superlevel filtration.
"""

from typing import Dict, List, Tuple

import gudhi
import networkx as nx
import numpy as np

from config import NOISE_THRESHOLD


def assign_flooding_filtration(graph: nx.Graph) -> nx.Graph:
    """
    Assign flooding filtration values to nodes and edges.

    Thick vessels flood first, so we use a superlevel-set filtration:
    - Node: f(v) = max avg_thickness among incident edges
    - Edge: f(e) = min(f(u), f(v))

    To use GUDHI's sublevel persistence, we negate all values so that
    thicker vessels get lower (earlier) filtration values.

    Args:
        graph: networkx graph with edge attr `avg_thickness`.

    Returns:
        Same graph with `flood_filt` attribute on nodes and edges.
    """
    # first pass: node filtration = max thickness of incident edges
    for node in graph.nodes():
        max_thick = 0.0
        for _, _, data in graph.edges(node, data=True):
            t = data.get("avg_thickness", 0.0)
            if t > max_thick:
                max_thick = t
        # negate for sublevel persistence (thick → low value → appears first)
        graph.nodes[node]["flood_filt"] = -max_thick

    # second pass: edge filtration = max(f(u), f(v))
    # since values are negated, max of negatives = min of originals
    for u, v in graph.edges():
        fu = graph.nodes[u]["flood_filt"]
        fv = graph.nodes[v]["flood_filt"]
        graph.edges[u, v]["flood_filt"] = max(fu, fv)

    return graph


def build_simplex_tree(graph: nx.Graph) -> gudhi.SimplexTree:
    """
    Build a GUDHI SimplexTree from flooding-annotated graph.

    Args:
        graph: networkx graph with `flood_filt` on all nodes and edges.

    Returns:
        Populated SimplexTree.
    """
    st = gudhi.SimplexTree()

    node_to_id = {}
    for i, (node, data) in enumerate(graph.nodes(data=True)):
        node_to_id[node] = i
        st.insert([i], filtration=data["flood_filt"])

    for u, v, data in graph.edges(data=True):
        uid = node_to_id[u]
        vid = node_to_id[v]
        st.insert([uid, vid], filtration=data["flood_filt"])

    return st


def _lifespans(intervals: List[Tuple[float, float]]) -> np.ndarray:
    """Compute lifespans from (birth, death) intervals, skipping infinites."""
    spans = []
    for b, d in intervals:
        if d == float("inf"):
            continue
        spans.append(abs(d - b))
    return np.array(spans) if spans else np.array([0.0])


def _persistence_entropy(intervals: List[Tuple[float, float]]) -> float:
    """Shannon entropy of the persistence diagram."""
    spans = _lifespans(intervals)
    spans = spans[spans > 0]

    if len(spans) == 0:
        return 0.0

    total = spans.sum()
    if total == 0:
        return 0.0

    probs = spans / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def extract_flooding_features(
    graph: nx.Graph,
    noise_threshold: float = NOISE_THRESHOLD,
) -> Dict[str, float]:
    """
    Full flooding filtration pipeline: assign values, build simplex tree,
    compute persistence, and extract TDA features.

    Features:
        flood_b0_max_lifespan     — max lifespan of H0 features
        flood_b0_sum_lifespan     — total lifespan of H0 features
        flood_b1_count            — number of H1 features above threshold
        flood_b1_max_lifespan     — max lifespan of H1 features
        flood_persistence_entropy — Shannon entropy of H1 diagram

    Args:
        graph:           networkx graph with edge attr `avg_thickness`.
        noise_threshold: minimum lifespan to count a 1-dim feature.

    Returns:
        Dict of feature name -> value.
    """
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return _empty_features()

    graph = assign_flooding_filtration(graph)
    st = build_simplex_tree(graph)
    persistence = st.persistence()

    h0 = [(b, d) for dim, (b, d) in persistence if dim == 0]
    h1 = [(b, d) for dim, (b, d) in persistence if dim == 1]

    h0_spans = _lifespans(h0)
    b0_max = float(h0_spans.max()) if len(h0_spans) > 0 else 0.0
    b0_sum = float(h0_spans.sum())

    h1_spans = _lifespans(h1)
    h1_significant = h1_spans[h1_spans > noise_threshold]

    b1_count = int(len(h1_significant))
    b1_max = float(h1_spans.max()) if len(h1_spans) > 0 else 0.0
    entropy = _persistence_entropy(h1)

    return {
        "flood_b0_max_lifespan": round(b0_max, 4),
        "flood_b0_sum_lifespan": round(b0_sum, 4),
        "flood_b1_count": b1_count,
        "flood_b1_max_lifespan": round(b1_max, 4),
        "flood_persistence_entropy": round(entropy, 6),
    }


def _empty_features() -> Dict[str, float]:
    return {
        "flood_b0_max_lifespan": 0.0,
        "flood_b0_sum_lifespan": 0.0,
        "flood_b1_count": 0,
        "flood_b1_max_lifespan": 0.0,
        "flood_persistence_entropy": 0.0,
    }
