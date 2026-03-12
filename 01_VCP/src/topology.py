"""
Topology estimation algorithms for Retinal Artery-Vein Classification.

Based on: "Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"
Reference: Section 2.3 (Vessel Topology Estimation), Algorithm 1

Key algorithms:
1. Tree tracing via connectivity prediction
2. Segment thickness computation
3. Final AV classification with voting
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from collections import defaultdict
import heapq

from .graph_utils import VesselGraph, extract_vessel_segments


class TopologyEstimator:
    """
    Vessel Topology Estimation using Connectivity Prediction.
    
    Reference: Section 2.3 - "we cast the vessel topology estimation as iterative 
    vascular connectivity prediction, which is specifically implemented as 
    deep-learning-based pairwise classification between vessel pixels across 
    a junction."
    
    The topology is estimated by:
    1. Computing connectivity probabilities between segment pairs
    2. Tree tracing from the thinnest terminal segment outward
    3. Assigning tree labels based on connectivity
    """
    
    def __init__(
        self,
        connectivity_threshold: float = 0.5,
        av_reassign_threshold: float = 0.8
    ):
        """
        Args:
            connectivity_threshold: Threshold for connectivity probability
            av_reassign_threshold: Threshold for AV reassignment (Section 2.4)
        """
        self.connectivity_threshold = connectivity_threshold
        self.av_reassign_threshold = av_reassign_threshold
    
    def compute_segment_thickness(
        self,
        graph: VesselGraph,
        thickness_map: np.ndarray
    ) -> np.ndarray:
        """
        Compute thickness for each segment.
        
        Reference: Equation 5 - "The thickness of the i-th segment is defined by:
        t_i = t(v_i) = (1/N_i) * Σ argmax T(v_im)"
        
        Args:
            graph: VesselGraph object
            thickness_map: Thickness probability map [C, H, W] or values [H, W]
            
        Returns:
            Array of thickness values for each segment
        """
        num_segments = len(graph.vertices)
        thicknesses = np.zeros(num_segments)
        
        H_t = thickness_map.shape[-2]
        W_t = thickness_map.shape[-1]

        for seg_idx, coords in enumerate(graph.vertices):
            if len(coords) == 0:
                continue
            ys, xs = coords[:, 0], coords[:, 1]
            valid = (ys >= 0) & (ys < H_t) & (xs >= 0) & (xs < W_t)
            ys, xs = ys[valid], xs[valid]
            if len(ys) == 0:
                continue
            if thickness_map.ndim == 3:
                t = np.argmax(thickness_map[:, ys, xs], axis=0)  # [N]
            else:
                t = thickness_map[ys, xs]  # [N]
            thicknesses[seg_idx] = t.mean()
        
        return thicknesses


    def compute_segment_connectivity_probs(
        self,
        connectivity_predictions: Dict[Tuple[int, int], float]
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute connectivity probability between each pair of segments.
        
        Reference: Equation 4 - "The connectivity probability between the j-th 
        and the k-th vessel segments is then computed by averaging the 
        probabilities of pixel pairs"
        
        P^conn_jk = P^conn(v_j, v_k) = (1/|S_jk|) * Σ P^conn(v_jm, v_kn | W)
        
        Args:
            connectivity_predictions: Dict mapping (seg_i, seg_j) to list of pixel pair probs
            
        Returns:
            Dict mapping segment pairs to aggregated connectivity probability
        """
        segment_probs = {}
        
        for (seg_i, seg_j), prob in connectivity_predictions.items():
            key = (min(seg_i, seg_j), max(seg_i, seg_j))
            if key not in segment_probs:
                segment_probs[key] = []
            segment_probs[key].append(prob)
        
        # Average probabilities
        for key in segment_probs:
            segment_probs[key] = np.mean(segment_probs[key])
        
        return segment_probs


    def tree_tracing(
        self,
        graph: VesselGraph,
        segment_connectivity: Dict[Tuple[int, int], float],
        segment_thicknesses: np.ndarray,
        optic_disc_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Tree tracing via connectivity prediction.
        
        Reference: Algorithm 1 - Tree tracing via connectivity prediction
        
        "While the farthest part from the optic disc center is firstly labeled in [9],
        we start the tracing from the thinnest terminal segment (segment with a 
        terminal pixel). Since vessels become thinner when being spread out, we 
        believe that the criterion based on the thickness is more appropriate to 
        choose the stable start point for tracing."
        
        Args:
            graph: VesselGraph object
            segment_connectivity: Connectivity probabilities between segment pairs
            segment_thicknesses: Thickness values for each segment
            optic_disc_mask: Optional mask of optic disc region
            
        Returns:
            Tree labels for each segment (0 = unassigned, 1+ = tree ID)
        """
        num_segments = len(graph.vertices)
        tree_labels = np.zeros(num_segments, dtype=np.int32)
        
        if num_segments == 0:
            return tree_labels
        
        # Find terminal segments (segments with only one connection)
        segment_degrees = defaultdict(int)
        for v1, v2 in graph.edges:
            segment_degrees[v1] += 1
            segment_degrees[v2] += 1
        
        terminal_segments = [
            i for i in range(num_segments)
            if segment_degrees[i] <= 1
        ]
        
        if not terminal_segments:
            # No terminal segments, use all segments
            terminal_segments = list(range(num_segments))
        
        # Initialize
        y_cntr = 0  # Tree counter

        # Sort terminal segments by thickness (thinnest first)
        terminal_segments.sort(key=lambda x: segment_thicknesses[x])

        # Precompute adjacency list from connectivity dict (avoids O(N_segs × |E|) per iteration)
        adjacency: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (v1, v2), prob in segment_connectivity.items():
            if prob >= self.connectivity_threshold:
                adjacency[v1].append((v2, prob))
                adjacency[v2].append((v1, prob))

        # Main loop
        while True:
            # Find untracked non-terminal segments with assigned tree label
            untracked_nonterminal = []
            for seg in range(num_segments):
                if tree_labels[seg] > 0:
                    for neighbor, prob in adjacency[seg]:
                        if tree_labels[neighbor] == 0:
                            untracked_nonterminal.append((seg, prob))
                            break

            if untracked_nonterminal:
                untracked_nonterminal.sort(key=lambda x: -x[1])
                current_seg = untracked_nonterminal[0][0]
            else:
                untracked_terminals = [seg for seg in terminal_segments if tree_labels[seg] == 0]
                if untracked_terminals:
                    untracked_terminals.sort(key=lambda x: segment_thicknesses[x])
                    y_cntr += 1
                    current_seg = untracked_terminals[0]
                    tree_labels[current_seg] = y_cntr
                else:
                    break

            current_tree = tree_labels[current_seg]
            for neighbor, _ in adjacency[current_seg]:
                if tree_labels[neighbor] == 0:
                    tree_labels[neighbor] = current_tree
        
        return tree_labels


    def classify_trees(
        self,
        graph: VesselGraph,
        tree_labels: np.ndarray,
        av_predictions: np.ndarray,
        vessel_mask: np.ndarray
    ) -> np.ndarray:
        """
        Classify each tree as artery or vein.
        
        Reference: Section 2.4 - "Each separate tree is then classified as an artery 
        or vein using a voting scheme where the probability vectors of all 
        constituent centerline pixels are averaged."
        
        Args:
            graph: VesselGraph object
            tree_labels: Tree label for each segment
            av_predictions: AV probability map [C, H, W] (C=3: bg, artery, vein)
            vessel_mask: Binary vessel mask
            
        Returns:
            Final AV classification [H, W] (0=bg, 1=artery, 2=vein)
        """
        H, W = vessel_mask.shape
        av_result = np.zeros((H, W), dtype=np.uint8)
        
        unique_trees = np.unique(tree_labels)
        unique_trees = unique_trees[unique_trees > 0]  # Exclude unassigned
        
        for tree_id in unique_trees:
            tree_segments = np.where(tree_labels == tree_id)[0]

            # Gather all pixel coordinates for this tree at once
            coord_list = [graph.vertices[s] for s in tree_segments if len(graph.vertices[s]) > 0]
            if not coord_list:
                continue
            all_coords = np.vstack(coord_list)
            ys, xs = all_coords[:, 0], all_coords[:, 1]
            valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            ys, xs = ys[valid], xs[valid]
            if len(ys) == 0:
                continue

            # Vectorized vote
            if av_predictions.ndim == 3:
                avg_artery = av_predictions[1, ys, xs].mean()
                avg_vein   = av_predictions[2, ys, xs].mean()
            else:
                avg_artery = (av_predictions[ys, xs] == 1).mean()
                avg_vein   = (av_predictions[ys, xs] == 2).mean()

            tree_class = 1 if avg_artery > avg_vein else 2
            prob = max(avg_artery, avg_vein)

            if prob > self.av_reassign_threshold:
                # Vectorized pixel assignment
                for seg_idx in tree_segments:
                    coords = graph.vertices[seg_idx]
                    syx = coords[:, 0]; sxx = coords[:, 1]
                    sv = (syx >= 0) & (syx < H) & (sxx >= 0) & (sxx < W)
                    av_result[syx[sv], sxx[sv]] = tree_class
        
        # For unassigned pixels, use pixel-wise prediction
        unassigned = (av_result == 0) & (vessel_mask > 0)
        if av_predictions.ndim == 3:
            pixelwise = np.argmax(av_predictions, axis=0)
        else:
            pixelwise = av_predictions
        av_result[unassigned] = pixelwise[unassigned]
        
        return av_result


def estimate_topology_and_classify(
    vessel_mask: np.ndarray,
    av_predictions: np.ndarray,
    thickness_map: np.ndarray,
    connectivity_fn,
    optic_disc_mask: Optional[np.ndarray] = None,
    connectivity_threshold: float = 0.5,
    av_reassign_threshold: float = 0.8
) -> Tuple[np.ndarray, VesselGraph, np.ndarray]:
    """
    Full topology estimation and AV classification pipeline.
    
    Reference: Figure 2 - Block diagram of the proposed method
    
    Args:
        vessel_mask: Binary vessel segmentation [H, W]
        av_predictions: AV classification predictions [C, H, W] or [H, W]
        thickness_map: Thickness map [C, H, W] or [H, W]
        connectivity_fn: Function to predict connectivity for a pixel pair
        optic_disc_mask: Optional optic disc mask
        connectivity_threshold: Threshold for connectivity
        av_reassign_threshold: Threshold for AV reassignment
        
    Returns:
        final_av: Final AV classification [H, W]
        graph: Vessel graph
        tree_labels: Tree labels for each segment
    """
    # Build vessel graph
    graph = extract_vessel_segments(vessel_mask)
    
    # Compute segment thicknesses
    estimator = TopologyEstimator(
        connectivity_threshold=connectivity_threshold,
        av_reassign_threshold=av_reassign_threshold
    )
    
    segment_thicknesses = estimator.compute_segment_thickness(graph, thickness_map)
    
    # Predict connectivity for all segment pairs
    connectivity_predictions = {}
    
    for (v1, v2) in graph.edges:
        # Get representative pixels from each segment
        if len(graph.vertices[v1]) > 0 and len(graph.vertices[v2]) > 0:
            # Sample multiple pairs and average
            num_samples = min(10, len(graph.vertices[v1]), len(graph.vertices[v2]))
            probs = []
            
            for _ in range(num_samples):
                idx1 = np.random.randint(len(graph.vertices[v1]))
                idx2 = np.random.randint(len(graph.vertices[v2]))
                
                coord1 = graph.vertices[v1][idx1]
                coord2 = graph.vertices[v2][idx2]
                
                prob = connectivity_fn(coord1, coord2)
                probs.append(prob)
            
            connectivity_predictions[(v1, v2)] = np.mean(probs)
    
    # Add potential edges for segments near junctions but not connected
    # (These are the "unconnected" cases)
    for i in range(len(graph.vertices)):
        for j in range(i + 1, len(graph.vertices)):
            if (i, j) not in connectivity_predictions and (j, i) not in connectivity_predictions:
                if len(graph.vertices[i]) > 0 and len(graph.vertices[j]) > 0:
                    # Check if segments are close
                    c1 = graph.vertices[i].mean(axis=0)
                    c2 = graph.vertices[j].mean(axis=0)
                    dist = np.linalg.norm(c1 - c2)
                    
                    if dist < 50:  # Only consider nearby segments
                        coord1 = graph.vertices[i][len(graph.vertices[i])//2]
                        coord2 = graph.vertices[j][len(graph.vertices[j])//2]
                        
                        prob = connectivity_fn(coord1, coord2)
                        connectivity_predictions[(i, j)] = prob
    
    # Tree tracing
    tree_labels = estimator.tree_tracing(
        graph,
        connectivity_predictions,
        segment_thicknesses,
        optic_disc_mask
    )
    
    # Final AV classification
    final_av = estimator.classify_trees(
        graph,
        tree_labels,
        av_predictions,
        vessel_mask
    )
    
    return final_av, graph, tree_labels


def visualize_topology(
    vessel_mask: np.ndarray,
    tree_labels: np.ndarray,
    graph: VesselGraph,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a visualization of the estimated topology.
    
    Different colors for different trees.
    
    Args:
        vessel_mask: Binary vessel mask
        tree_labels: Tree labels for each segment
        graph: VesselGraph object
        output_path: Optional path to save the visualization
        
    Returns:
        RGB visualization image
    """
    H, W = vessel_mask.shape
    
    # Create colormap for different trees
    unique_trees = np.unique(tree_labels)
    num_trees = len(unique_trees[unique_trees > 0])
    
    # Generate colors using HSV
    colors = []
    for i in range(num_trees + 1):
        if i == 0:
            colors.append((0, 0, 0))  # Background
        else:
            hue = (i - 1) / max(num_trees, 1)
            # Convert HSV to RGB (simplified)
            r = int(255 * abs(np.sin(hue * np.pi * 2)))
            g = int(255 * abs(np.sin(hue * np.pi * 2 + np.pi / 3)))
            b = int(255 * abs(np.sin(hue * np.pi * 2 + 2 * np.pi / 3)))
            colors.append((r, g, b))
    
    # Create RGB image
    result = np.zeros((H, W, 3), dtype=np.uint8)
    color_array = np.array(colors, dtype=np.uint8)  # [num_colors, 3]

    for seg_idx, tree_label in enumerate(tree_labels):
        if tree_label > 0 and seg_idx < len(graph.vertices):
            coords = graph.vertices[seg_idx]
            ys, xs = coords[:, 0], coords[:, 1]
            valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            color_idx = min(tree_label, len(color_array) - 1)
            result[ys[valid], xs[valid]] = color_array[color_idx]
    
    if output_path:
        from PIL import Image
        Image.fromarray(result).save(output_path)
    
    return result
