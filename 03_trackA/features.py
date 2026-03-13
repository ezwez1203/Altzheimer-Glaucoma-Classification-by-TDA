"""
features.py — TDA feature extraction from persistence diagrams.
"""

from typing import Dict, List, Tuple

import numpy as np

from config import NOISE_THRESHOLD


def _lifespans(intervals: List[Tuple[float, float]]) -> np.ndarray:
    """Compute lifespans from a list of (birth, death) intervals."""
    spans = []
    for b, d in intervals:
        if d == float("inf"):
            continue
        spans.append(d - b)
    return np.array(spans) if spans else np.array([0.0])


def _persistence_entropy(intervals: List[Tuple[float, float]]) -> float:
    """
    Shannon entropy of the persistence diagram.

    Normalizes lifespans into a probability distribution, then computes:
        H = -sum(p_i * log(p_i))

    Args:
        intervals: list of (birth, death) pairs.

    Returns:
        Entropy value. 0.0 if there are no finite intervals.
    """
    spans = _lifespans(intervals)
    spans = spans[spans > 0]

    if len(spans) == 0:
        return 0.0

    total = spans.sum()
    if total == 0:
        return 0.0

    probs = spans / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def extract_tda_features(
    persistence: list,
    noise_threshold: float = NOISE_THRESHOLD,
) -> Dict[str, float]:
    """
    Compute TDA feature vector from persistence output.

    Features:
        b0_max_lifespan     — max lifespan among 0-dim features
        b0_sum_lifespan     — total lifespan of 0-dim features
        b1_count            — number of 1-dim features above noise threshold
        b1_max_lifespan     — max lifespan among 1-dim features
        persistence_entropy — Shannon entropy of the 1-dim diagram

    Args:
        persistence:     output of simplex_tree.persistence().
        noise_threshold: minimum lifespan to count as a real feature.

    Returns:
        Dict of feature name -> value.
    """
    # separate by dimension
    h0 = [(b, d) for dim, (b, d) in persistence if dim == 0]
    h1 = [(b, d) for dim, (b, d) in persistence if dim == 1]

    # H0 features
    h0_spans = _lifespans(h0)
    b0_max = float(h0_spans.max()) if len(h0_spans) > 0 else 0.0
    b0_sum = float(h0_spans.sum())

    # H1 features
    h1_spans = _lifespans(h1)
    h1_significant = h1_spans[h1_spans > noise_threshold]

    b1_count = int(len(h1_significant))
    b1_max = float(h1_spans.max()) if len(h1_spans) > 0 else 0.0
    entropy = _persistence_entropy(h1)

    return {
        "b0_max_lifespan": round(b0_max, 4),
        "b0_sum_lifespan": round(b0_sum, 4),
        "b1_count": b1_count,
        "b1_max_lifespan": round(b1_max, 4),
        "persistence_entropy": round(entropy, 6),
    }
