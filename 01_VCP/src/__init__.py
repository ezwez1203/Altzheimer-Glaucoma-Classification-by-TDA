"""
Topology-Aware Retinal Artery-Vein Classification via Deep Vascular Connectivity Prediction

Source code package implementing the paper methodology.

Modules:
    - dataset: PyTorch Dataset classes for retinal image loading
    - models: VGG16-based network architectures (OD, Multi-task, Connectivity)
    - preprocessing: Thickness/Orientation map generation with Voronoi tessellation
    - losses: Loss functions (Cross-entropy, Multi-task, Connectivity)
    - graph_utils: Vessel graph construction and junction detection
    - topology: Tree tracing algorithm for topology estimation
"""

from .dataset import RetinalDataset, ConnectivityDataset, PairwiseFeatureDataset
from .models import (
    VGG16Encoder,
    OpticDiscSegmentationNetwork,
    MultiTaskNetwork,
    ConnectivityNetwork,
    FullConnectivityPipeline,
    ThicknessOrientationEncoder,
    HadamardProductLayer
)
from .preprocessing import (
    generate_thickness_orientation_maps,
    compute_centerline,
    compute_thickness_map,
    compute_orientation_vectors,
    voronoi_propagation,
    quantize_thickness,
    quantize_orientation
)
from .losses import (
    DiceLoss,
    FocalLoss,
    WeightedCrossEntropyLoss,
    BinaryCrossEntropyLoss,
    MultiTaskLoss,
    ConnectivityLoss,
    OpticDiscLoss,
    CombinedLoss
)
from .graph_utils import (
    VesselGraph,
    detect_junctions,
    detect_endpoints,
    segment_centerline,
    extract_vessel_segments,
    get_segment_pairs_at_junction,
    generate_pixel_pairs_for_connectivity,
    refine_av_with_vessels
)
from .topology import (
    TopologyEstimator,
    estimate_topology_and_classify,
    visualize_topology
)

__all__ = [
    # Dataset
    'RetinalDataset',
    'ConnectivityDataset',
    'PairwiseFeatureDataset',
    
    # Models
    'VGG16Encoder',
    'OpticDiscSegmentationNetwork',
    'MultiTaskNetwork',
    'ConnectivityNetwork',
    'FullConnectivityPipeline',
    'ThicknessOrientationEncoder',
    'HadamardProductLayer',
    
    # Preprocessing
    'generate_thickness_orientation_maps',
    'compute_centerline',
    'compute_thickness_map',
    'compute_orientation_vectors',
    'voronoi_propagation',
    'quantize_thickness',
    'quantize_orientation',
    
    # Losses
    'DiceLoss',
    'FocalLoss',
    'WeightedCrossEntropyLoss',
    'BinaryCrossEntropyLoss',
    'MultiTaskLoss',
    'ConnectivityLoss',
    'OpticDiscLoss',
    'CombinedLoss',
    
    # Graph utilities
    'VesselGraph',
    'detect_junctions',
    'detect_endpoints',
    'segment_centerline',
    'extract_vessel_segments',
    'get_segment_pairs_at_junction',
    'generate_pixel_pairs_for_connectivity',
    'refine_av_with_vessels',
    
    # Topology
    'TopologyEstimator',
    'estimate_topology_and_classify',
    'visualize_topology'
]

__version__ = '1.0.0'
