"""
extractor.py — AlzheimerTDAExtractor: orchestrates all Track B biomarkers.
"""

from typing import Dict, Optional

import networkx as nx
import numpy as np

from fractal import fractal_dimension
from avr import compute_avr
from flooding import extract_flooding_features
from config import NOISE_THRESHOLD


class AlzheimerTDAExtractor:
    """
    Extracts three AD-relevant biomarkers from retinal vascular data:

        1. Fractal Dimension  (box-counting on vessel mask)
        2. AVR                (artery/vein thickness ratio from graph)
        3. Flooding TDA       (persistent homology on thickness-based filtration)
    """

    def __init__(self, noise_threshold: float = NOISE_THRESHOLD):
        self.noise_threshold = noise_threshold

    def extract_features(
        self,
        graph: nx.Graph,
        vessel_mask: np.ndarray,
        av_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Run all three biomarker extractions.

        Args:
            graph:       networkx graph from graph extraction stage.
            vessel_mask: (H, W) binary vessel mask (0/255).
            av_mask:     (H, W, 3) RGB AV mask (unused here; AVR comes from graph).

        Returns:
            Dict of all Track B features.
        """
        features = {}

        # biomarker 1: fractal dimension
        features["fractal_dimension"] = round(fractal_dimension(vessel_mask), 6)

        # biomarker 2: artery-vein ratio
        features["AVR"] = round(compute_avr(graph), 6)

        # biomarker 3: flooding filtration TDA
        flood_feats = extract_flooding_features(graph, self.noise_threshold)
        features.update(flood_feats)

        return features
