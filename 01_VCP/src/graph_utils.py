"""
Graph utilities for Topology-Aware Retinal Artery-Vein Classification.

Based on: "Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"
Reference: Section 2.2 (Graph Construction)

Key algorithms:
1. Graph construction from vessel segmentation
2. Junction detection
3. Vessel segment extraction
4. Segment pair generation for connectivity training
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from scipy import ndimage
from scipy.ndimage import label as ndimage_label
from skimage.morphology import skeletonize, remove_small_objects
from collections import defaultdict


class VesselGraph:
    """
    Graph representation of the vessel structure.
    
    Reference: Section 2.2 - "An initial graph G(V, E) is constructed as an 
    undirected graph. V = {v_j}^N_{j=1} is a set of vertices, and is defined 
    by setting each vessel segment as a vertex."
    
    Attributes:
        vertices: List of vessel segments (each segment is a set of pixel coords)
        edges: Set of edges (pairs of vertex indices)
        junctions: List of junction point coordinates
    """
    
    def __init__(self):
        self.vertices: List[np.ndarray] = []  # Each vertex is array of (y, x) coords
        self.edges: Set[Tuple[int, int]] = set()
        self.junctions: List[Tuple[int, int]] = []
        self.segment_labels: Optional[np.ndarray] = None
        self._adj: Optional[Dict] = None  # lazy adjacency cache

    def add_vertex(self, segment_coords: np.ndarray) -> int:
        """Add a vessel segment as a vertex."""
        self.vertices.append(segment_coords)
        self._adj = None  # invalidate cache
        return len(self.vertices) - 1

    def add_edge(self, v1: int, v2: int):
        """Add an edge between two vertices."""
        if v1 != v2:
            self.edges.add((min(v1, v2), max(v1, v2)))
            self._adj = None  # invalidate cache

    def _build_adj(self):
        """Build adjacency list once and cache it."""
        adj: Dict[int, List[int]] = defaultdict(list)
        for v1, v2 in self.edges:
            adj[v1].append(v2)
            adj[v2].append(v1)
        self._adj = adj

    def get_neighbors(self, vertex_idx: int) -> List[int]:
        """Get neighboring vertices — O(1) after first call."""
        if self._adj is None:
            self._build_adj()
        return self._adj.get(vertex_idx, [])

    def get_degree(self, vertex_idx: int) -> int:
        """Get the degree (number of connections) of a vertex."""
        return len(self.get_neighbors(vertex_idx))


def detect_junctions(centerline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect junction points on the vessel centerline.
    
    Reference: Section 2.2 - "junction detection by locating intersection points 
    on the centerlines"
    
    A junction is a pixel with more than 2 neighbors on the centerline.
    
    Args:
        centerline: Binary centerline mask [H, W]
        
    Returns:
        junction_mask: Binary mask of junction points
        junction_coords: Array of (y, x) coordinates of junctions
    """
    # Define 8-connectivity kernel
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    
    # Count neighbors for each pixel
    neighbor_count = ndimage.convolve(
        centerline.astype(np.uint8),
        kernel,
        mode='constant',
        cval=0
    )
    
    # Junction points have more than 2 neighbors
    # Also must be on the centerline
    junction_mask = (neighbor_count > 2) & (centerline > 0)
    
    # Get coordinates
    junction_coords = np.array(np.where(junction_mask)).T
    
    return junction_mask.astype(np.uint8), junction_coords


def detect_endpoints(centerline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect endpoint (terminal) pixels on the vessel centerline.
    
    Endpoints have exactly 1 neighbor on the centerline.
    
    Args:
        centerline: Binary centerline mask [H, W]
        
    Returns:
        endpoint_mask: Binary mask of endpoints
        endpoint_coords: Array of (y, x) coordinates of endpoints
    """
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    
    neighbor_count = ndimage.convolve(
        centerline.astype(np.uint8),
        kernel,
        mode='constant',
        cval=0
    )
    
    endpoint_mask = (neighbor_count == 1) & (centerline > 0)
    endpoint_coords = np.array(np.where(endpoint_mask)).T
    
    return endpoint_mask.astype(np.uint8), endpoint_coords


def segment_centerline(
    centerline: np.ndarray,
    junction_mask: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Segment the centerline into separate vessel segments.
    
    Reference: Section 2.2 - "vessel segment separation by removing the detected 
    junction points"
    
    Removes junction points and labels connected components to get separate segments.
    
    Args:
        centerline: Binary centerline mask [H, W]
        junction_mask: Binary junction mask [H, W]
        
    Returns:
        segment_labels: Label image where each segment has a unique label
        num_segments: Number of segments
    """
    # Remove junction points to separate segments
    # Dilate junctions slightly to ensure complete separation
    junction_dilated = ndimage.binary_dilation(
        junction_mask,
        iterations=1
    )
    
    # Remove junctions from centerline
    separated = centerline.astype(bool) & ~junction_dilated
    
    # Label connected components
    segment_labels, num_segments = ndimage_label(
        separated,
        structure=np.ones((3, 3))  # 8-connectivity
    )
    
    return segment_labels, num_segments


def extract_vessel_segments(
    vessel_mask: np.ndarray,
    min_segment_length: int = 10
) -> VesselGraph:
    """
    Extract vessel graph from a binary vessel mask.
    
    Reference: Section 2.2 - Graph Construction
    
    Steps:
    1. Skeletonize to get centerline
    2. Detect junctions
    3. Segment centerline by removing junctions
    4. Build graph with segments as vertices and junctions as edge connections
    
    Args:
        vessel_mask: Binary vessel segmentation mask [H, W]
        min_segment_length: Minimum number of pixels for a valid segment
        
    Returns:
        VesselGraph object
    """
    # Ensure binary
    mask = (vessel_mask > 0).astype(np.uint8)
    
    # Skeletonize
    centerline = skeletonize(mask).astype(np.uint8)
    
    # Detect junctions
    junction_mask, junction_coords = detect_junctions(centerline)
    
    # Segment centerline
    segment_labels, num_segments = segment_centerline(centerline, junction_mask)
    
    # Build graph
    graph = VesselGraph()
    graph.junctions = [tuple(coord) for coord in junction_coords]
    graph.segment_labels = segment_labels
    
    # Add segments as vertices
    for seg_id in range(1, num_segments + 1):
        seg_coords = np.array(np.where(segment_labels == seg_id)).T
        
        # Skip small segments
        if len(seg_coords) < min_segment_length:
            continue
        
        graph.add_vertex(seg_coords)
    
    # Build edges based on junction connectivity
    # For each junction pixel, find which graph vertices touch it (vectorized)
    junction_to_segments = defaultdict(set)
    H_jm, W_jm = junction_mask.shape

    # Build a fast lookup: pixel -> graph vertex index
    # graph.vertices[i] are the pixel coords for vertex i (1-indexed segment_labels → vertex index)
    pixel_to_vertex = np.full((H_jm, W_jm), -1, dtype=np.int32)
    for seg_idx, seg_coords in enumerate(graph.vertices):
        pixel_to_vertex[seg_coords[:, 0], seg_coords[:, 1]] = seg_idx

    # For each junction pixel, scan 8-neighbors in pixel_to_vertex
    junc_ys, junc_xs = junction_coords[:, 0], junction_coords[:, 1]
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        ny = np.clip(junc_ys + dy, 0, H_jm - 1)
        nx = np.clip(junc_xs + dx, 0, W_jm - 1)
        nbr_vertex = pixel_to_vertex[ny, nx]  # [N_junctions]
        valid = nbr_vertex >= 0
        for jy, jx, vi in zip(junc_ys[valid], junc_xs[valid], nbr_vertex[valid]):
            junction_to_segments[(int(jy), int(jx))].add(int(vi))
    
    # Add edges
    for junction, segments in junction_to_segments.items():
        segments = list(segments)
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                graph.add_edge(segments[i], segments[j])
    
    return graph


def get_segment_pairs_at_junction(
    graph: VesselGraph,
    max_distance: int = 10
) -> List[Tuple[int, int, bool]]:
    """
    Get pairs of segments that meet at junctions.
    
    Reference: Section 2.3.1 - "The probability P^conn(v_jm, v_kn|W) is defined 
    between a pixel v_jm on the j-th segment and another pixel v_kn on the k-th 
    segment."
    
    Args:
        graph: VesselGraph object
        max_distance: Maximum distance for segment pair consideration
        
    Returns:
        List of (segment_i, segment_j, is_connected) tuples
    """
    pairs = []
    
    for v1, v2 in graph.edges:
        # Connected pairs
        pairs.append((v1, v2, True))
    
    # Add some non-connected pairs (negative samples)
    num_vertices = len(graph.vertices)
    for v1 in range(num_vertices):
        for v2 in range(v1 + 1, num_vertices):
            if (v1, v2) not in graph.edges and (v2, v1) not in graph.edges:
                # Check if they're within distance
                # Use centroid distance
                c1 = graph.vertices[v1].mean(axis=0)
                c2 = graph.vertices[v2].mean(axis=0)
                dist = np.sqrt(np.sum((c1 - c2) ** 2))
                
                if dist < max_distance * 5:  # Nearby but not connected
                    pairs.append((v1, v2, False))
    
    return pairs


def generate_pixel_pairs_for_connectivity(
    graph: VesselGraph,
    num_pairs_per_junction: int = 100,
    distance_threshold: int = 10
) -> List[Dict]:
    """
    Generate pixel pairs for connectivity training.
    
    Reference: Section 2.1.4 - "For any two pixels that are respectively on 
    different vessel segments joining at a junction, features extracted at 
    their image coordinates are fed into the fc layers."
    
    Args:
        graph: VesselGraph object
        num_pairs_per_junction: Number of pairs to sample per junction
        distance_threshold: Maximum distance from junction for sampling
        
    Returns:
        List of dictionaries with pixel pair information
    """
    pairs = []
    
    segment_pairs = get_segment_pairs_at_junction(graph)
    
    for seg_i, seg_j, is_connected in segment_pairs:
        # Get coordinates for each segment
        coords_i = graph.vertices[seg_i]
        coords_j = graph.vertices[seg_j]
        
        if len(coords_i) == 0 or len(coords_j) == 0:
            continue
        
        # Sample pairs
        num_i = len(coords_i)
        num_j = len(coords_j)
        
        for _ in range(min(num_pairs_per_junction, num_i * num_j)):
            idx_i = np.random.randint(0, num_i)
            idx_j = np.random.randint(0, num_j)
            
            pairs.append({
                'coords_i': coords_i[idx_i].tolist(),
                'coords_j': coords_j[idx_j].tolist(),
                'segment_i': seg_i,
                'segment_j': seg_j,
                'connected': int(is_connected)
            })
    
    return pairs


def refine_av_with_vessels(
    av_prediction: np.ndarray,
    vessel_mask: np.ndarray
) -> np.ndarray:
    """
    Refine AV prediction by multiplying with vessel segmentation.
    
    Reference: Section 2.2 - "Before the graph construction, the AV classification 
    result is simply refined by multiplying with the binary vessel segmentation."
    
    Args:
        av_prediction: AV classification result [H, W]
        vessel_mask: Binary vessel mask [H, W]
        
    Returns:
        Refined AV prediction
    """
    return av_prediction * (vessel_mask > 0).astype(av_prediction.dtype)


def extract_av_centerlines(
    av_prediction: np.ndarray,
    artery_label: int = 1,
    vein_label: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract centerlines for arteries and veins separately.
    
    Reference: Section 2.2 - "centerline extraction by morphological thinning 
    for arteries and veins"
    
    Args:
        av_prediction: AV classification result [H, W]
        artery_label: Label for arteries
        vein_label: Label for veins
        
    Returns:
        artery_centerline, vein_centerline
    """
    artery_mask = (av_prediction == artery_label).astype(np.uint8)
    vein_mask = (av_prediction == vein_label).astype(np.uint8)
    
    artery_centerline = skeletonize(artery_mask).astype(np.uint8)
    vein_centerline = skeletonize(vein_mask).astype(np.uint8)
    
    return artery_centerline, vein_centerline


def create_segment_label_map(
    graph: VesselGraph,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Create a label map where each pixel is assigned to its segment.
    
    Args:
        graph: VesselGraph object
        image_shape: (H, W) shape of the output
        
    Returns:
        Label map [H, W] where each segment has a unique ID
    """
    label_map = np.zeros(image_shape, dtype=np.int32)
    H, W = image_shape

    for seg_idx, coords in enumerate(graph.vertices):
        ys, xs = coords[:, 0], coords[:, 1]
        valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        label_map[ys[valid], xs[valid]] = seg_idx + 1  # 0 is background

    return label_map


def propagate_segment_labels_to_vessel(
    segment_label_map: np.ndarray,
    vessel_mask: np.ndarray
) -> np.ndarray:
    """
    Propagate segment labels from centerline to full vessel mask.
    
    Uses Voronoi-like propagation (nearest neighbor).
    
    Args:
        segment_label_map: Label map on centerline [H, W]
        vessel_mask: Binary vessel mask [H, W]
        
    Returns:
        Propagated label map [H, W]
    """
    from scipy.ndimage import distance_transform_edt
    
    # For each vessel pixel, find nearest labeled centerline pixel
    labeled = segment_label_map > 0
    
    if not np.any(labeled):
        return np.zeros_like(segment_label_map)
    
    # Distance transform from unlabeled to labeled
    _, indices = distance_transform_edt(~labeled, return_indices=True)
    
    # Propagate labels
    propagated = segment_label_map[indices[0], indices[1]]
    
    # Mask to vessel only
    propagated = propagated * (vessel_mask > 0).astype(propagated.dtype)
    
    return propagated
