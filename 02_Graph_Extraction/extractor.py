"""
extractor.py — RetinalGraphExtractor: orchestrates the full extraction pipeline.
"""

import networkx as nx
import numpy as np

from config import MIN_EDGE_LENGTH
from morphology import (
    compute_thickness_map,
    compute_skeleton,
    count_neighbors,
    detect_nodes,
)
from tracer import trace_edges
from graph_builder import build_av_label_map, build_graph


class RetinalGraphExtractor:
    """
    Extracts a vascular connectivity graph from retinal segmentation masks.

    Pipeline:
        1. Distance transform  -> thickness map
        2. Skeletonization     -> centerline
        3. Node detection      -> endpoints + junctions
        4. Edge tracing        -> weighted edges with morphometric attributes
    """

    def __init__(self, min_edge_length: int = MIN_EDGE_LENGTH):
        """
        Args:
            min_edge_length: discard edges shorter than this (removes noise spurs).
        """
        self.min_edge_length = min_edge_length

    def extract_graph(
        self,
        vessel_mask: np.ndarray,
        av_mask: np.ndarray,
    ) -> nx.Graph:
        """
        Run the full extraction pipeline on a single image pair.

        Args:
            vessel_mask: (H, W) binary uint8 mask (0 or 255).
            av_mask:     (H, W, 3) RGB image — red=artery, blue=vein.

        Returns:
            networkx.Graph with node/edge attributes.
        """
        binary = (vessel_mask > 0).astype(np.uint8)

        # step 1 — thickness
        thickness_map = compute_thickness_map(binary)

        # step 2 — skeleton
        skeleton = compute_skeleton(binary)

        # step 3 — nodes
        neighbor_cnt = count_neighbors(skeleton)
        endpoints, junctions = detect_nodes(skeleton, neighbor_cnt)
        node_set = endpoints | junctions

        # step 4 — edges
        edges = trace_edges(skeleton, node_set, self.min_edge_length)

        # step 5 — graph assembly
        av_label = build_av_label_map(av_mask)
        graph = build_graph(edges, endpoints, junctions, thickness_map, av_label)

        return graph
