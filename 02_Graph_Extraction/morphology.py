"""
morphology.py — Distance transform, skeletonization, and node detection.
"""

from typing import Set, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from config import NEIGHBORS_8


def compute_thickness_map(binary_mask: np.ndarray) -> np.ndarray:
    """
    Euclidean distance transform * 2 to estimate vessel diameter.

    Args:
        binary_mask: (H, W) binary array (1 = vessel, 0 = background).

    Returns:
        (H, W) float array of estimated vessel thickness.
    """
    return distance_transform_edt(binary_mask) * 2.0


def compute_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization to 1-pixel centerline.

    Args:
        binary_mask: (H, W) binary array.

    Returns:
        (H, W) uint8 skeleton (0 or 1).
    """
    return skeletonize(binary_mask).astype(np.uint8)


def count_neighbors(skeleton: np.ndarray) -> np.ndarray:
    """
    Count 8-connected skeleton neighbors for every pixel.

    Args:
        skeleton: (H, W) uint8 skeleton image.

    Returns:
        (H, W) int32 array of neighbor counts.
    """
    h, w = skeleton.shape
    count = np.zeros_like(skeleton, dtype=np.int32)
    for dy, dx in NEIGHBORS_8:
        shifted = np.zeros_like(skeleton)
        sy = slice(max(0, -dy), h - max(0, dy))
        sx = slice(max(0, -dx), w - max(0, dx))
        ty = slice(max(0, dy), h - max(0, -dy))
        tx = slice(max(0, dx), w - max(0, -dx))
        shifted[ty, tx] = skeleton[sy, sx]
        count += shifted
    return count


def detect_nodes(
    skeleton: np.ndarray,
    neighbor_count: np.ndarray,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Classify skeleton pixels into endpoints and junctions.

    Args:
        skeleton:       (H, W) uint8 skeleton.
        neighbor_count: (H, W) int32 neighbor counts.

    Returns:
        (endpoints, junctions) — each a set of (y, x) tuples.
    """
    endpoints = set(zip(*np.where((skeleton == 1) & (neighbor_count == 1))))
    junctions = set(zip(*np.where((skeleton == 1) & (neighbor_count >= 3))))
    return endpoints, junctions
