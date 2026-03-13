"""
fractal.py — Box-counting fractal dimension of binary vessel masks.
"""

from typing import List

import numpy as np
from scipy.stats import linregress

from config import BOX_SIZES


def box_count(mask: np.ndarray, box_size: int) -> int:
    """
    Count the number of non-empty boxes of a given size covering the mask.

    Args:
        mask:     (H, W) binary array (1 = vessel).
        box_size: side length of each box in pixels.

    Returns:
        Number of boxes that contain at least one vessel pixel.
    """
    h, w = mask.shape
    count = 0
    for y in range(0, h, box_size):
        for x in range(0, w, box_size):
            patch = mask[y:y + box_size, x:x + box_size]
            if patch.any():
                count += 1
    return count


def fractal_dimension(
    vessel_mask: np.ndarray,
    box_sizes: List[int] = None,
) -> float:
    """
    Estimate fractal dimension via 2D box-counting.

    Fits log(N(s)) vs log(1/s) with linear regression and returns
    the slope as the fractal dimension D.

    Args:
        vessel_mask: (H, W) binary uint8 mask (0/255 or 0/1).
        box_sizes:   list of box side lengths to test.

    Returns:
        Estimated fractal dimension (typically 1.0 ~ 2.0 for vessels).
        Returns 0.0 if the mask is empty.
    """
    if box_sizes is None:
        box_sizes = BOX_SIZES

    binary = (vessel_mask > 0).astype(np.uint8)

    if not binary.any():
        return 0.0

    counts = []
    sizes = []
    for s in box_sizes:
        n = box_count(binary, s)
        if n > 0:
            counts.append(n)
            sizes.append(s)

    if len(sizes) < 2:
        return 0.0

    log_inv_s = np.log(1.0 / np.array(sizes))
    log_n = np.log(np.array(counts))

    slope, _, _, _, _ = linregress(log_inv_s, log_n)
    return float(slope)
