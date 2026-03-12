"""
Inference script for Topology-Aware Retinal Artery-Vein Classification.

Based on: "Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"
Reference: Figure 2 (Block diagram of the proposed method at test time)

Inference Pipeline:
1. Pixelwise prediction on optic disc and vessel
2. Graph construction from vessel segmentation and AV classification
3. Connectivity prediction between vessel segments
4. Topology estimation via tree tracing
5. Final AV classification with voting
"""

import os
import sys
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.models import (
    OpticDiscSegmentationNetwork,
    MultiTaskNetwork,
    FullConnectivityPipeline
)
from src.preprocessing import generate_thickness_orientation_maps
from src.graph_utils import (
    extract_vessel_segments,
    VesselGraph
)
from src.topology import (
    TopologyEstimator,
    estimate_topology_and_classify,
    visualize_topology
)


class RetinalAVClassifier:
    """
    Full inference pipeline for Retinal Artery-Vein Classification.
    
    This class loads all trained models and provides methods for:
    - Optic disc segmentation
    - Binary vessel segmentation
    - Pixelwise AV classification
    - Topology-aware AV classification (tree-wise)
    """
    
    def __init__(
        self,
        config: Config,
        device: str = 'cuda',
        od_checkpoint: Optional[str] = None,
        multitask_checkpoint: Optional[str] = None,
        connectivity_checkpoint: Optional[str] = None
    ):
        """
        Args:
            config: Configuration object
            device: 'cuda' or 'cpu'
            od_checkpoint: Path to optic disc segmentation checkpoint
            multitask_checkpoint: Path to multi-task network checkpoint
            connectivity_checkpoint: Path to connectivity network checkpoint
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.od_model = None
        self.multitask_model = None
        self.connectivity_model = None
        
        # Load checkpoints
        if od_checkpoint and os.path.exists(od_checkpoint):
            self._load_od_model(od_checkpoint)
        
        if multitask_checkpoint and os.path.exists(multitask_checkpoint):
            self._load_multitask_model(multitask_checkpoint)
        
        if connectivity_checkpoint and os.path.exists(connectivity_checkpoint):
            self._load_connectivity_model(connectivity_checkpoint)
        
        # Topology estimator
        self.topology_estimator = TopologyEstimator(
            connectivity_threshold=config.CONNECTIVITY_THRESHOLD,
            av_reassign_threshold=config.AV_REASSIGN_THRESHOLD
        )
        
        # Image preprocessing
        self.image_size = config.IMAGE_SIZE
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
    
    def _load_od_model(self, checkpoint_path: str):
        """Load optic disc segmentation model."""
        self.od_model = OpticDiscSegmentationNetwork(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.od_model.load_state_dict(checkpoint['model_state_dict'])
        self.od_model = self.od_model.to(self.device)
        self.od_model.eval()
        print(f"Loaded OD model from {checkpoint_path}")
    
    def _load_multitask_model(self, checkpoint_path: str):
        """Load multi-task network."""
        self.multitask_model = MultiTaskNetwork(
            pretrained=False,
            num_av_classes=self.config.NUM_AV_CLASSES,
            multiscale_channels=self.config.MULTISCALE_CHANNELS
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.multitask_model.load_state_dict(checkpoint['model_state_dict'])
        self.multitask_model = self.multitask_model.to(self.device)
        self.multitask_model.eval()
        print(f"Loaded multi-task model from {checkpoint_path}")
    
    def _load_connectivity_model(self, checkpoint_path: str):
        """Load connectivity classification network."""
        self.connectivity_model = FullConnectivityPipeline(
            pretrained=False,
            multiscale_channels=self.config.MULTISCALE_CHANNELS,
            num_thickness_classes=self.config.NUM_THICKNESS_CLASSES,
            num_orientation_classes=self.config.NUM_ORIENTATION_CLASSES,
            fc_units=self.config.FC_UNITS
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.connectivity_model.load_state_dict(checkpoint['model_state_dict'])
        self.connectivity_model = self.connectivity_model.to(self.device)
        self.connectivity_model.eval()
        print(f"Loaded connectivity model from {checkpoint_path}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: RGB image [H, W, 3] or grayscale [H, W]
            
        Returns:
            Preprocessed tensor [1, 3, H', W']
        """
        # Convert to RGB if grayscale
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Resize
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((self.image_size[1], self.image_size[0]))
        
        # To tensor
        tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        tensor = tensor.to(self.device)
        
        # Normalize
        tensor = (tensor - self.mean) / self.std
        
        return tensor
    
    @torch.no_grad()
    def predict_optic_disc(self, image: np.ndarray) -> np.ndarray:
        """
        Predict optic disc segmentation.
        
        Args:
            image: RGB image [H, W, 3]
            
        Returns:
            Binary OD mask [H, W]
        """
        if self.od_model is None:
            raise RuntimeError("OD model not loaded")
        
        original_size = image.shape[:2]
        tensor = self.preprocess_image(image)
        
        logits = self.od_model(tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        
        # Resize to original
        pred = F.interpolate(pred, size=original_size, mode='nearest')
        
        return pred.squeeze().cpu().numpy().astype(np.uint8)
    
    @torch.no_grad()
    def predict_vessel_and_av(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict vessel segmentation and AV classification.
        
        Args:
            image: RGB image [H, W, 3]
            
        Returns:
            vessel_mask: Binary vessel mask [H, W]
            av_pred: AV class predictions [H, W] (0=bg, 1=artery, 2=vein)
            av_probs: AV probabilities [3, H, W]
        """
        if self.multitask_model is None:
            raise RuntimeError("Multi-task model not loaded")
        
        original_size = image.shape[:2]
        tensor = self.preprocess_image(image)
        
        vessel_logits, av_logits, _ = self.multitask_model(tensor)
        
        # Vessel prediction
        vessel_probs = torch.sigmoid(vessel_logits)
        vessel_pred = (vessel_probs > 0.5).float()
        
        # AV prediction
        av_probs = F.softmax(av_logits, dim=1)
        av_pred = torch.argmax(av_probs, dim=1)
        
        # Resize to original
        vessel_pred = F.interpolate(vessel_pred, size=original_size, mode='nearest')
        av_probs = F.interpolate(av_probs, size=original_size, mode='bilinear', align_corners=True)
        av_pred = F.interpolate(av_pred.unsqueeze(1).float(), size=original_size, mode='nearest')
        
        return (
            vessel_pred.squeeze().cpu().numpy().astype(np.uint8),
            av_pred.squeeze().cpu().numpy().astype(np.uint8),
            av_probs.squeeze().cpu().numpy()
        )
    
    @torch.no_grad()
    def predict_connectivity(
        self,
        image: np.ndarray,
        coord_i: np.ndarray,
        coord_j: np.ndarray
    ) -> float:
        """
        Predict connectivity between two pixel coordinates.
        
        Args:
            image: RGB image [H, W, 3]
            coord_i: (y, x) coordinate of first pixel
            coord_j: (y, x) coordinate of second pixel
            
        Returns:
            Connectivity probability
        """
        if self.connectivity_model is None:
            raise RuntimeError("Connectivity model not loaded")
        
        tensor = self.preprocess_image(image)
        
        # Scale coordinates to model input size
        scale_y = self.image_size[0] / image.shape[0]
        scale_x = self.image_size[1] / image.shape[1]
        
        coord_i_scaled = torch.tensor([
            [coord_i[0] * scale_y, coord_i[1] * scale_x]
        ], device=self.device).float()
        
        coord_j_scaled = torch.tensor([
            [coord_j[0] * scale_y, coord_j[1] * scale_x]
        ], device=self.device).float()
        
        _, _, conn_logits = self.connectivity_model(tensor, coord_i_scaled, coord_j_scaled)
        
        if conn_logits is not None:
            conn_probs = F.softmax(conn_logits, dim=1)
            return conn_probs[0, 1].cpu().item()  # Probability of "connected"
        
        return 0.5
    
    def classify_with_topology(
        self,
        image: np.ndarray,
        vessel_mask: Optional[np.ndarray] = None,
        av_probs: Optional[np.ndarray] = None,
        od_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Full topology-aware AV classification pipeline.
        
        Reference: Figure 2 - Complete test-time pipeline
        
        Args:
            image: RGB image [H, W, 3]
            vessel_mask: Optional precomputed vessel mask
            av_probs: Optional precomputed AV probabilities
            od_mask: Optional precomputed OD mask
            
        Returns:
            Dictionary containing all predictions
        """
        results = {}
        
        # Step 1: Optic disc segmentation
        if od_mask is None and self.od_model is not None:
            od_mask = self.predict_optic_disc(image)
        results['od_mask'] = od_mask
        
        # Step 2: Vessel segmentation and pixelwise AV
        if vessel_mask is None or av_probs is None:
            vessel_mask, av_pred, av_probs = self.predict_vessel_and_av(image)
        else:
            av_pred = np.argmax(av_probs, axis=0)
        
        results['vessel_mask'] = vessel_mask
        # Paper: "AV classification result is simply refined by multiplying with 
        # the binary vessel segmentation"
        av_pred_masked = av_pred * (vessel_mask > 0).astype(np.uint8)
        results['av_pixelwise'] = av_pred_masked
        results['av_probs'] = av_probs
        
        # Exclude OD region from vessels
        if od_mask is not None:
            vessel_mask = vessel_mask * (1 - od_mask)
        
        # Step 3: Build vessel graph
        graph = extract_vessel_segments(vessel_mask)
        results['num_segments'] = len(graph.vertices)
        
        # Step 4: Generate thickness/orientation maps
        thick_cont, ori_cont, _, _ = generate_thickness_orientation_maps(vessel_mask)
        results['thickness_map'] = thick_cont
        results['orientation_map'] = ori_cont
        
        # Step 5: Topology estimation with connectivity prediction
        if self.connectivity_model is not None:
            def connectivity_fn(coord_i, coord_j):
                return self.predict_connectivity(image, coord_i, coord_j)
            
            final_av, graph, tree_labels = estimate_topology_and_classify(
                vessel_mask,
                av_probs,
                thick_cont,
                connectivity_fn,
                od_mask,
                self.config.CONNECTIVITY_THRESHOLD,
                self.config.AV_REASSIGN_THRESHOLD
            )
            
            results['av_treewise'] = final_av
            results['tree_labels'] = tree_labels
            
            # Visualize topology
            topology_vis = visualize_topology(vessel_mask, tree_labels, graph)
            results['topology_visualization'] = topology_vis
        else:
            # Fallback to pixelwise
            results['av_treewise'] = av_pred
        
        return results


def process_single_image(
    image_path: str,
    classifier: RetinalAVClassifier,
    output_dir: str
):
    """Process a single image and save results."""
    
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"Processing: {base_name}")
    
    # Run classification
    results = classifier.classify_with_topology(image)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save vessel mask
    Image.fromarray(results['vessel_mask'] * 255).save(
        os.path.join(output_dir, f'{base_name}_vessel.png')
    )
    
    # Save AV predictions (color coded)
    av_vis = np.zeros((*results['av_treewise'].shape, 3), dtype=np.uint8)
    av_vis[results['av_treewise'] == 1] = [255, 0, 0]  # Artery = Red
    av_vis[results['av_treewise'] == 2] = [0, 0, 255]  # Vein = Blue
    Image.fromarray(av_vis).save(
        os.path.join(output_dir, f'{base_name}_av_treewise.png')
    )
    
    # Save pixelwise AV
    av_pix_vis = np.zeros((*results['av_pixelwise'].shape, 3), dtype=np.uint8)
    av_pix_vis[results['av_pixelwise'] == 1] = [255, 0, 0]
    av_pix_vis[results['av_pixelwise'] == 2] = [0, 0, 255]
    Image.fromarray(av_pix_vis).save(
        os.path.join(output_dir, f'{base_name}_av_pixelwise.png')
    )
    
    # Save topology visualization
    if 'topology_visualization' in results:
        Image.fromarray(results['topology_visualization']).save(
            os.path.join(output_dir, f'{base_name}_topology.png')
        )
    
    # Save OD mask
    if results['od_mask'] is not None:
        Image.fromarray(results['od_mask'] * 255).save(
            os.path.join(output_dir, f'{base_name}_od.png')
        )
    
    # Save thickness map (color-coded)
    if 'thickness_map' in results and results['thickness_map'] is not None:
        thickness = results['thickness_map']
        # Normalize to 0-255
        thickness_norm = thickness.copy()
        if thickness_norm.max() > 0:
            thickness_norm = (thickness_norm / thickness_norm.max() * 255).astype(np.uint8)
        # Apply colormap (thicker vessels = brighter)
        thickness_vis = np.zeros((*thickness.shape, 3), dtype=np.uint8)
        thickness_vis[..., 0] = thickness_norm  # Red channel for thickness
        thickness_vis[..., 1] = thickness_norm // 2  # Some green
        thickness_vis[..., 2] = 0  # No blue
        Image.fromarray(thickness_vis).save(
            os.path.join(output_dir, f'{base_name}_thickness.png')
        )
    
    # Save orientation map (color-coded by angle)
    if 'orientation_map' in results and results['orientation_map'] is not None:
        orientation = results['orientation_map']
        # Map orientation angles to colors (HSV-like)
        # Orientation ranges from -pi/2 to pi/2
        orientation_vis = np.zeros((*orientation.shape, 3), dtype=np.uint8)
        valid = np.abs(orientation) > 0  # Only where there are vessels
        if valid.any():
            # Normalize to 0-1
            ori_norm = (orientation[valid] + np.pi/2) / np.pi
            # Map to RGB using angle as hue
            orientation_vis[valid, 0] = (np.sin(ori_norm * np.pi) * 255).astype(np.uint8)
            orientation_vis[valid, 1] = (np.sin(ori_norm * np.pi + np.pi/3) * 255).astype(np.uint8)
            orientation_vis[valid, 2] = (np.sin(ori_norm * np.pi + 2*np.pi/3) * 255).astype(np.uint8)
        Image.fromarray(orientation_vis).save(
            os.path.join(output_dir, f'{base_name}_orientation.png')
        )
    
    print(f"  Segments: {results['num_segments']}")
    print(f"  Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Inference for Topology-Aware Retinal AV Classification'
    )
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('--output', '-o', default='results',
                        help='Output directory')
    parser.add_argument('--dataset', type=str, default='DRIVE',
                        choices=['DRIVE', 'IOSTAR'],
                        help='Dataset checkpoints to use')
    parser.add_argument('--od-checkpoint', type=str, default=None,
                        help='Path to OD model checkpoint')
    parser.add_argument('--multitask-checkpoint', type=str, default=None,
                        help='Path to multi-task model checkpoint')
    parser.add_argument('--connectivity-checkpoint', type=str, default=None,
                        help='Path to connectivity model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Set default checkpoint paths
    if args.od_checkpoint is None:
        args.od_checkpoint = os.path.join(
            config.CHECKPOINT_DIR, f'od_segm_{args.dataset}_best.pth'
        )
    if args.multitask_checkpoint is None:
        args.multitask_checkpoint = os.path.join(
            config.CHECKPOINT_DIR, f'multitask_{args.dataset}_best.pth'
        )
    if args.connectivity_checkpoint is None:
        args.connectivity_checkpoint = os.path.join(
            config.CHECKPOINT_DIR, f'connectivity_{args.dataset}_best.pth'
        )
    
    # Create classifier
    classifier = RetinalAVClassifier(
        config,
        device=args.device,
        od_checkpoint=args.od_checkpoint if os.path.exists(args.od_checkpoint) else None,
        multitask_checkpoint=args.multitask_checkpoint if os.path.exists(args.multitask_checkpoint) else None,
        connectivity_checkpoint=args.connectivity_checkpoint if os.path.exists(args.connectivity_checkpoint) else None
    )
    
    # Process input
    if os.path.isfile(args.input):
        process_single_image(args.input, classifier, args.output)
    elif os.path.isdir(args.input):
        # Process all images in directory
        import glob
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(args.input, ext)))
        
        for img_path in tqdm(sorted(image_files), desc="Processing images"):
            process_single_image(img_path, classifier, args.output)
    else:
        print(f"Input not found: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
