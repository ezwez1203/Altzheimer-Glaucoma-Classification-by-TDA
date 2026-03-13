"""
extractor.py — RadialFiltrationTDA: orchestrates the full TDA pipeline.
"""

from typing import Dict, Optional

import networkx as nx
import numpy as np

from filtration import assign_filtration_values, build_simplex_tree, compute_od_center
from features import extract_tda_features


class RadialFiltrationTDA:
    """
    Radial filtration-based TDA feature extractor for retinal vascular graphs.

    Pipeline:
        1. Compute OD center from the optic disc mask
        2. Assign radial filtration values (distance from OD center)
        3. Build GUDHI SimplexTree
        4. Compute persistent homology
        5. Extract TDA feature vector
    """

    def __init__(self, noise_threshold: float = 5.0):
        """
        Args:
            noise_threshold: minimum lifespan to count a 1-dim feature as significant.
        """
        self.noise_threshold = noise_threshold

    def extract_features(
        self,
        graph: nx.Graph,
        od_mask: Optional[np.ndarray] = None,
        image_shape: tuple = (584, 565),
    ) -> Dict[str, float]:
        """
        Run the full TDA pipeline on a single graph.

        Args:
            graph:       networkx graph from graph extraction (nodes have y, x attrs).
            od_mask:     (H, W) binary OD mask. If None, falls back to image center.
            image_shape: (H, W) used as fallback center when od_mask is unavailable.

        Returns:
            Dict of TDA feature name -> value.
        """
        if graph.number_of_nodes() == 0:
            return self._empty_features()

        # step 1 — OD center
        od_center = compute_od_center(od_mask)
        if od_center is None:
            od_center = (image_shape[0] / 2.0, image_shape[1] / 2.0)

        # step 2 — assign filtration values
        graph = assign_filtration_values(graph, od_center)

        # step 3 — simplex tree
        st = build_simplex_tree(graph)

        # step 4 — persistent homology
        persistence = st.persistence()

        # step 5 — features
        return extract_tda_features(persistence, self.noise_threshold)

    @staticmethod
    def _empty_features() -> Dict[str, float]:
        """Return zeroed feature dict for empty graphs."""
        return {
            "b0_max_lifespan": 0.0,
            "b0_sum_lifespan": 0.0,
            "b1_count": 0,
            "b1_max_lifespan": 0.0,
            "persistence_entropy": 0.0,
        }
