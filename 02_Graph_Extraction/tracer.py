"""
tracer.py — Edge tracing along skeleton branches between nodes.
"""

from typing import List, Set, Tuple

import numpy as np

from config import NEIGHBORS_8


def trace_edges(
    skeleton: np.ndarray,
    node_set: Set[Tuple[int, int]],
    min_edge_length: int = 3,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]]:
    """
    Walk along skeleton branches between detected nodes (endpoints / junctions).

    For each node, explore every 8-connected neighbor and follow the skeleton
    until another node is reached, recording the full pixel path.

    Args:
        skeleton:         (H, W) uint8 skeleton image.
        node_set:         set of (y, x) coordinates that are nodes.
        min_edge_length:  discard edges shorter than this to filter noise.

    Returns:
        List of (start_node, end_node, pixel_path) tuples.
    """
    h, w = skeleton.shape
    visited_edges: set = set()
    edges = []

    for start in node_set:
        for dy, dx in NEIGHBORS_8:
            ny, nx_ = start[0] + dy, start[1] + dx
            if not (0 <= ny < h and 0 <= nx_ < w):
                continue
            if skeleton[ny, nx_] == 0:
                continue

            # walk from start through (ny, nx_) until hitting another node
            path = [start, (ny, nx_)]
            prev = start
            curr = (ny, nx_)

            while curr not in node_set:
                next_pixel = None
                for dy2, dx2 in NEIGHBORS_8:
                    nn = (curr[0] + dy2, curr[1] + dx2)
                    if nn == prev:
                        continue
                    if not (0 <= nn[0] < h and 0 <= nn[1] < w):
                        continue
                    if skeleton[nn[0], nn[1]] == 1:
                        next_pixel = nn
                        break

                if next_pixel is None:
                    break

                path.append(next_pixel)
                prev = curr
                curr = next_pixel

            end = curr
            if end not in node_set:
                continue

            edge_key = frozenset((start, end))
            if edge_key in visited_edges and start != end:
                continue
            if start == end and (start, end) in visited_edges:
                continue

            if len(path) < min_edge_length:
                continue

            visited_edges.add(edge_key)
            if start == end:
                visited_edges.add((start, end))
            edges.append((start, end, path))

    return edges
