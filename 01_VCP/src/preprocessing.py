"""
Preprocessing utilities for Topology-Aware Retinal Artery-Vein Classification.

Based on: "Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"
Reference: Section 2.1.3 (Vascular Thickness/Orientation Classification)

Key algorithms:
1. Thickness computation using distance transform on vessel centerlines
2. Orientation computation using centerline direction vectors
3. Voronoi tessellation for propagating values to all vessel pixels
"""

from typing import Tuple, Optional

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage.measure import label


def compute_centerline(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute the centerline (skeleton) of a binary vessel mask.
    
    Reference: Section 2.1.3 - "For each pixel on vessel centerlines, 
    the distance value is doubled and recorded."
    
    Args:
        binary_mask: Binary vessel mask [H, W], values 0 or 1
        
    Returns:
        centerline: Binary centerline mask [H, W]
    """
    # Ensure binary
    mask = (binary_mask > 0).astype(np.uint8)
    
    # Skeletonize to get centerline
    centerline = skeletonize(mask).astype(np.uint8)
    
    return centerline


def compute_thickness_map(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute vessel thickness map using distance transform.
    
    Reference: Section 2.1.3 - "As for the thickness, a distance transform is 
    firstly applied to the segmentation GT for computing distances to the closest 
    vessel boundary. Then, for each pixel on vessel centerlines, the distance 
    value is doubled and recorded."
    
    The thickness at each centerline pixel is 2 * the distance to the boundary,
    which approximates the vessel diameter.
    
    Args:
        binary_mask: Binary vessel mask [H, W]
        
    Returns:
        thickness_map: Thickness values at centerline pixels [H, W]
                      Non-centerline pixels are 0
    """
    # Ensure binary
    mask = (binary_mask > 0).astype(np.uint8)
    
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    
    # Compute distance transform (distance to nearest background pixel)
    distance = distance_transform_edt(mask)
    
    # Get centerline
    centerline = compute_centerline(mask)
    
    # Thickness = 2 * distance at centerline pixels
    thickness_map = np.zeros_like(mask, dtype=np.float32)
    centerline_coords = np.where(centerline > 0)
    thickness_map[centerline_coords] = 2.0 * distance[centerline_coords]
    
    return thickness_map


def compute_orientation_vectors(centerline: np.ndarray, segment_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orientation vectors on vessel centerlines.
    
    Reference: Section 2.1.3 - "As for the orientation, vessel centerlines are 
    partitioned into segments of a certain length (five pixels here). The angle 
    between the x-axis and the straight line connecting the end points of each 
    segment is computed in a range of [-π/2, π/2]."
    
    Args:
        centerline: Binary centerline mask [H, W]
        segment_length: Length of centerline segments for orientation computation
        
    Returns:
        orientation_map: Orientation angles in radians [-π/2, π/2] at centerline pixels
        orientation_valid: Boolean mask indicating valid orientation values
    """
    H, W = centerline.shape
    orientation_map = np.zeros((H, W), dtype=np.float32)
    orientation_valid = np.zeros((H, W), dtype=bool)
    
    if centerline.sum() == 0:
        return orientation_map, orientation_valid
    
    # Label connected components of the centerline
    labeled_centerline, num_components = label(centerline, connectivity=2, return_num=True)
    
    for comp_id in range(1, num_components + 1):
        # Get pixels of this component
        comp_mask = (labeled_centerline == comp_id)
        coords = np.array(np.where(comp_mask)).T  # [N, 2] where each row is (y, x)
        
        if len(coords) < 2:
            continue
        
        # Order pixels along the centerline using nearest neighbor traversal
        ordered_coords = _order_centerline_coords(coords)
        
        # Compute orientation for segments
        for i in range(len(ordered_coords)):
            # Find segment endpoints
            start_idx = max(0, i - segment_length // 2)
            end_idx = min(len(ordered_coords) - 1, i + segment_length // 2)
            
            if end_idx <= start_idx:
                continue
            
            y1, x1 = ordered_coords[start_idx]
            y2, x2 = ordered_coords[end_idx]
            
            # Compute angle with x-axis
            dy = y2 - y1
            dx = x2 - x1
            
            if dx == 0 and dy == 0:
                continue
            
            # atan2 gives angle in [-π, π], we want [-π/2, π/2]
            angle = np.arctan2(dy, dx)
            
            # Normalize to [-π/2, π/2]
            while angle > np.pi / 2:
                angle -= np.pi
            while angle < -np.pi / 2:
                angle += np.pi
            
            y, x = ordered_coords[i]
            orientation_map[y, x] = angle
            orientation_valid[y, x] = True
    
    return orientation_map, orientation_valid


def _order_centerline_coords(coords: np.ndarray) -> np.ndarray:
    """
    Order centerline coordinates using nearest neighbor traversal.

    Vectorized with numpy: each step computes all distances at once,
    avoiding the O(N²) pure-Python loop.

    Args:
        coords: Unordered coordinates [N, 2]

    Returns:
        ordered_coords: Ordered coordinates [N, 2]
    """
    n = len(coords)
    if n <= 2:
        return coords

    visited = np.zeros(n, dtype=bool)
    ordered = np.empty_like(coords)
    ordered[0] = coords[0]
    visited[0] = True

    for i in range(1, n):
        current = ordered[i - 1]
        diff = coords - current                          # [N, 2]
        dists = diff[:, 0] ** 2 + diff[:, 1] ** 2      # [N]  (squared, no sqrt needed)
        dists[visited] = np.iinfo(np.int64).max         # mask already-visited
        nearest_idx = int(dists.argmin())
        if dists[nearest_idx] > 8:                      # disconnected jump — pick any unvisited
            nearest_idx = int(np.where(~visited)[0][0])
        ordered[i] = coords[nearest_idx]
        visited[nearest_idx] = True

    return ordered


def voronoi_propagation(
    values: np.ndarray,
    valid_mask: np.ndarray,
    vessel_mask: np.ndarray
) -> np.ndarray:
    """
    Propagate values from centerline to all vessel pixels using Voronoi tessellation
    (nearest neighbor assignment).
    
    Reference: Section 2.1.3 - "The computed thickness/orientation values are 
    respectively assigned to nearby vessel pixels using Voronoi tessellation."
    
    This effectively assigns each vessel pixel the value of its nearest centerline pixel.
    
    Args:
        values: Values at centerline pixels [H, W]
        valid_mask: Boolean mask of valid (centerline) pixels [H, W]
        vessel_mask: Binary vessel mask [H, W]
        
    Returns:
        propagated: Values propagated to all vessel pixels [H, W]
    """
    H, W = values.shape
    
    vessel_binary = (vessel_mask > 0)
    valid_binary = valid_mask.astype(bool) & vessel_binary
    
    if not np.any(valid_binary):
        return np.zeros_like(values)
    
    # Compute distance transform from valid pixels
    # distance_transform_edt returns the distance to the nearest 0 pixel
    # So we invert: distance from background to foreground (valid pixels)
    invalid_mask = ~valid_binary
    
    # For each non-valid vessel pixel, find the nearest valid pixel
    # We use the feature transform (indices)
    _, indices = ndimage.distance_transform_edt(invalid_mask, return_indices=True)
    
    # Propagate values
    propagated = np.zeros((H, W), dtype=np.float32)
    
    # For valid pixels, copy the values directly
    propagated[valid_binary] = values[valid_binary]
    
    # For other vessel pixels, use nearest neighbor
    vessel_not_valid = vessel_binary & ~valid_binary
    if np.any(vessel_not_valid):
        y_coords = indices[0][vessel_not_valid]
        x_coords = indices[1][vessel_not_valid]
        propagated[vessel_not_valid] = values[y_coords, x_coords]
    
    # Zero out non-vessel pixels
    propagated[~vessel_binary] = 0
    
    return propagated


def quantize_thickness(
    thickness: np.ndarray,
    boundaries: list = [1.5, 3.0, 5.0, 7.0]
) -> np.ndarray:
    """
    Quantize thickness values into discrete classes.
    
    Reference: Section 3.2 - "For the thickness, we made five classes using 
    quantization boundaries of [1.5, 3, 5, 7]."
    
    Classes:
        0: thickness <= 1.5
        1: 1.5 < thickness <= 3.0
        2: 3.0 < thickness <= 5.0
        3: 5.0 < thickness <= 7.0
        4: thickness > 7.0
    
    Args:
        thickness: Continuous thickness values [H, W]
        boundaries: List of boundary values
        
    Returns:
        quantized: Class labels [H, W], dtype=int
    """
    quantized = np.zeros_like(thickness, dtype=np.int64)
    
    for i, boundary in enumerate(boundaries):
        quantized[thickness > boundary] = i + 1
    
    return quantized


def quantize_orientation(
    orientation: np.ndarray,
    num_classes: int = 6,
    valid_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Quantize orientation angles into discrete classes.
    
    Reference: Section 3.2 - "For the orientation task, every pixel was classified 
    into one of seven classes, including six classes evenly dividing the range of 
    [-π/2, π/2] and one background class."
    
    Classes:
        0: Background (non-vessel)
        1-6: Six evenly-spaced orientation bins in [-π/2, π/2]
    
    Args:
        orientation: Orientation angles in radians [-π/2, π/2]
        num_classes: Number of orientation classes (excluding background)
        valid_mask: Boolean mask of valid (vessel) pixels
        
    Returns:
        quantized: Class labels [H, W], dtype=int
    """
    # Normalize to [0, 1]
    normalized = (orientation + np.pi / 2) / np.pi
    
    # Clip to valid range
    normalized = np.clip(normalized, 0, 1 - 1e-6)
    
    # Quantize to class bins (1 to num_classes)
    quantized = (normalized * num_classes).astype(np.int64) + 1
    
    # Set background pixels to class 0
    if valid_mask is not None:
        quantized[~valid_mask] = 0
    
    return quantized


def generate_thickness_orientation_maps(
    binary_mask: np.ndarray,
    thickness_boundaries: list = [1.5, 3.0, 5.0, 7.0],
    num_orientation_classes: int = 6,
    segment_length: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate thickness and orientation maps from a binary vessel mask.
    
    Reference: Section 2.1.3 (Vascular Thickness/Orientation Classification)
    
    Pipeline:
    1. Compute vessel centerline using skeletonization
    2. Compute thickness at centerline pixels using distance transform
    3. Compute orientation angles at centerline pixels using segment direction
    4. Propagate values to all vessel pixels using Voronoi tessellation
    5. Quantize values into discrete classes for classification training
    
    Args:
        binary_mask: Binary vessel segmentation mask [H, W]
        thickness_boundaries: Quantization boundaries for thickness
        num_orientation_classes: Number of orientation classes
        segment_length: Length of centerline segments for orientation
        
    Returns:
        thickness_continuous: Continuous thickness values [H, W]
        orientation_continuous: Continuous orientation values [H, W]
        thickness_quantized: Quantized thickness classes [H, W]
        orientation_quantized: Quantized orientation classes [H, W]
    """
    # Ensure binary
    vessel_mask = (binary_mask > 0).astype(np.uint8)
    
    # Step 1: Compute centerline
    centerline = compute_centerline(vessel_mask)
    
    # Step 2: Compute thickness at centerlines
    thickness_centerline = compute_thickness_map(vessel_mask)
    
    # Step 3: Compute orientation at centerlines
    orientation_centerline, orientation_valid = compute_orientation_vectors(
        centerline, segment_length=segment_length
    )
    
    # Step 4: Propagate to all vessel pixels using Voronoi tessellation
    thickness_valid = centerline.astype(bool)
    thickness_continuous = voronoi_propagation(
        thickness_centerline, thickness_valid, vessel_mask
    )
    
    orientation_continuous = voronoi_propagation(
        orientation_centerline, orientation_valid, vessel_mask
    )
    
    # Step 5: Quantize
    thickness_quantized = quantize_thickness(
        thickness_continuous, boundaries=thickness_boundaries
    )
    
    orientation_quantized = quantize_orientation(
        orientation_continuous,
        num_classes=num_orientation_classes,
        valid_mask=vessel_mask.astype(bool)
    )
    
    return (
        thickness_continuous,
        orientation_continuous,
        thickness_quantized,
        orientation_quantized
    )


def generate_and_save_maps(
    vessel_mask_path: str,
    output_dir: str,
    thickness_boundaries: list = [1.5, 3.0, 5.0, 7.0],
    num_orientation_classes: int = 6
) -> None:
    """
    Generate thickness/orientation maps for a vessel mask and save to disk.
    
    Args:
        vessel_mask_path: Path to binary vessel mask image
        output_dir: Directory to save output maps
        thickness_boundaries: Quantization boundaries
        num_orientation_classes: Number of orientation classes
    """
    import os
    from PIL import Image
    
    # Load vessel mask
    mask = np.array(Image.open(vessel_mask_path).convert('L'))
    mask = (mask > 127).astype(np.uint8)
    
    # Generate maps
    thick_cont, ori_cont, thick_quant, ori_quant = generate_thickness_orientation_maps(
        mask, thickness_boundaries, num_orientation_classes
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(vessel_mask_path))[0]
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, f'{base_name}_thickness.npy'), thick_cont)
    np.save(os.path.join(output_dir, f'{base_name}_orientation.npy'), ori_cont)
    np.save(os.path.join(output_dir, f'{base_name}_thickness_quant.npy'), thick_quant)
    np.save(os.path.join(output_dir, f'{base_name}_orientation_quant.npy'), ori_quant)
    
    # Save visualizations
    thick_vis = (thick_cont / thick_cont.max() * 255).astype(np.uint8) if thick_cont.max() > 0 else np.zeros_like(thick_cont, dtype=np.uint8)
    Image.fromarray(thick_vis).save(os.path.join(output_dir, f'{base_name}_thickness_vis.png'))
    
    ori_vis = ((ori_cont + np.pi/2) / np.pi * 255).astype(np.uint8)
    ori_vis[mask == 0] = 0
    Image.fromarray(ori_vis).save(os.path.join(output_dir, f'{base_name}_orientation_vis.png'))


def batch_generate_maps(
    vessel_mask_dir: str,
    output_dir: str,
    thickness_boundaries: list = [1.5, 3.0, 5.0, 7.0],
    num_orientation_classes: int = 6,
    num_workers: int = None
) -> None:
    """
    Generate thickness/orientation maps for all vessel masks in a directory.
    Uses multiprocessing to process multiple files in parallel.

    Args:
        vessel_mask_dir: Directory containing binary vessel mask images
        output_dir: Directory to save output maps
        thickness_boundaries: Quantization boundaries
        num_orientation_classes: Number of orientation classes
        num_workers: Number of parallel workers (default: all CPU cores)
    """
    import os
    import glob
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    from functools import partial

    # Find all mask files
    extensions = ['*.png', '*.jpg', '*.tif', '*.gif', '*.bmp']
    mask_files = []
    for ext in extensions:
        mask_files.extend(glob.glob(os.path.join(vessel_mask_dir, ext)))

    mask_files = sorted(mask_files)
    print(f"Found {len(mask_files)} vessel masks")

    if num_workers is None:
        num_workers = cpu_count()

    worker_fn = partial(
        generate_and_save_maps,
        output_dir=output_dir,
        thickness_boundaries=thickness_boundaries,
        num_orientation_classes=num_orientation_classes
    )

    if num_workers > 1 and len(mask_files) > 1:
        print(f"Processing with {num_workers} parallel workers...")
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(worker_fn, mask_files), total=len(mask_files), desc="Generating maps"))
    else:
        for mask_path in tqdm(mask_files, desc="Generating maps"):
            generate_and_save_maps(mask_path, output_dir, thickness_boundaries, num_orientation_classes)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate thickness/orientation maps")
    parser.add_argument("input", help="Input vessel mask image or directory")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--thickness-boundaries", nargs="+", type=float,
                        default=[1.5, 3.0, 5.0, 7.0])
    parser.add_argument("--num-orientation-classes", type=int, default=6)
    
    args = parser.parse_args()
    
    import os
    
    if os.path.isdir(args.input):
        batch_generate_maps(
            args.input,
            args.output,
            args.thickness_boundaries,
            args.num_orientation_classes
        )
    else:
        generate_and_save_maps(
            args.input,
            args.output,
            args.thickness_boundaries,
            args.num_orientation_classes
        )
