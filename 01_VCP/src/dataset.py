"""
Dataset classes for Retinal Artery-Vein Classification.

Based on: "Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"
Reference: Section 3.1 (Dataset) and standard DRIVE/IOSTAR folder structures
"""

import os
import glob
from typing import Tuple, Optional, Dict, List, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RetinalDataset(Dataset):
    """
    PyTorch Dataset for loading retinal vessel images and labels.
    
    Supports DRIVE and IOSTAR dataset formats.
    
    Expected folder structure:
        root_dir/
        ├── training/
        │   ├── images/           # RGB fundus images
        │   ├── 1st_manual/       # Vessel segmentation ground truth
        │   ├── mask/             # FOV masks
        │   └── av/               # Optional: AV classification labels
        └── test/
            ├── images/
            ├── 1st_manual/
            ├── mask/
            └── av/
    
    Args:
        root_dir: Root directory of the dataset (e.g., 'data/DRIVE')
        mode: 'train' or 'test'
        image_size: Tuple of (height, width) for resizing
        augment: Whether to apply data augmentation
        return_av: Whether to return AV classification labels (if available)
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        return_av: bool = False,
        return_od: bool = False,
        thickness_dir: Optional[str] = None,
        orientation_dir: Optional[str] = None
    ):
        super().__init__()
        
        assert mode in ['train', 'test'], f"Mode must be 'train' or 'test', got {mode}"
        
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.augment = augment
        self.return_av = return_av
        self.return_od = return_od
        self.thickness_dir = thickness_dir
        self.orientation_dir = orientation_dir
        
        # Determine the subdirectory based on mode
        if mode == 'train':
            data_dir = os.path.join(root_dir, 'training')
        else:
            data_dir = os.path.join(root_dir, 'test')
        
        # Load file paths
        self.image_paths = self._load_files(os.path.join(data_dir, 'images'))
        self.vessel_paths = self._load_files(os.path.join(data_dir, '1st_manual'))
        self.mask_paths = self._load_files(os.path.join(data_dir, 'mask'))
        
        # Optional AV labels
        av_dir = os.path.join(data_dir, 'av')
        if os.path.exists(av_dir) and return_av:
            self.av_paths = self._load_files(av_dir)
        else:
            self.av_paths = None
            
        # Optional OD labels
        od_dir = os.path.join(data_dir, 'od')
        if os.path.exists(od_dir) and return_od:
            self.od_paths = self._load_files(od_dir)
        else:
            self.od_paths = None
        
        # Validate alignment
        assert len(self.image_paths) == len(self.vessel_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.vessel_paths)} vessel masks"
        
        if len(self.mask_paths) > 0:
            assert len(self.image_paths) == len(self.mask_paths), \
                f"Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} FOV masks"
        
        print(f"Loaded {len(self.image_paths)} samples from {data_dir}")

        # Pre-cache thickness/orientation paths to avoid per-sample os.path.exists() calls
        self._thickness_paths = self._resolve_aux_paths(self.vessel_paths, thickness_dir, 'thickness')
        self._orientation_paths = self._resolve_aux_paths(self.vessel_paths, orientation_dir, 'orientation')

        # Define transforms
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)
        
        # Normalization (ImageNet stats for pretrained VGG)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _resolve_aux_paths(self, vessel_paths: List[str], aux_dir: Optional[str], key: str) -> List[Optional[str]]:
        """Pre-compute auxiliary (thickness/orientation) .npy file paths at init time."""
        resolved = []
        for vpath in vessel_paths:
            if aux_dir is None:
                resolved.append(None)
                continue
            base = os.path.splitext(os.path.basename(vpath))[0]
            quant_path = os.path.join(aux_dir, f"{base}_{key}_quant.npy")
            raw_path = os.path.join(aux_dir, f"{base}_{key}.npy")
            if os.path.exists(quant_path):
                resolved.append(quant_path)
            elif os.path.exists(raw_path):
                resolved.append(raw_path)
            else:
                resolved.append(None)
        return resolved

    def _load_files(self, directory: str) -> List[str]:
        """Load all image files from a directory."""
        if not os.path.exists(directory):
            return []
        
        # Common image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.gif', '*.ppm']
        
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))
            files.extend(glob.glob(os.path.join(directory, ext.upper())))
        
        # Sort to ensure consistent ordering
        files = sorted(files)
        
        return files
    
    def _load_image(self, path: str, mode: str = 'RGB') -> Image.Image:
        """Load an image from path."""
        img = Image.open(path)
        if mode == 'RGB':
            img = img.convert('RGB')
        elif mode == 'L':
            img = img.convert('L')
        return img
    
    def _apply_augmentation(
        self,
        image: torch.Tensor,
        vessel: torch.Tensor,
        mask: torch.Tensor,
        av: Optional[torch.Tensor] = None,
        od: Optional[torch.Tensor] = None,
        thickness: Optional[torch.Tensor] = None,
        orientation: Optional[torch.Tensor] = None
    ) -> Tuple:
        """Apply synchronized data augmentation to image and masks."""
        
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            vessel = TF.hflip(vessel)
            mask = TF.hflip(mask)
            if av is not None:
                av = TF.hflip(av)
            if od is not None:
                od = TF.hflip(od)
            if thickness is not None:
                thickness = TF.hflip(thickness)
            if orientation is not None:
                orientation = TF.hflip(orientation)
        
        # Random rotation (-10 to 10 degrees)
        if torch.rand(1).item() > 0.5:
            angle = (torch.rand(1).item() - 0.5) * 20  # -10 to 10 degrees
            image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
            vessel = TF.rotate(vessel, angle, interpolation=T.InterpolationMode.NEAREST)
            mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
            if av is not None:
                av = TF.rotate(av, angle, interpolation=T.InterpolationMode.NEAREST)
            if od is not None:
                od = TF.rotate(od, angle, interpolation=T.InterpolationMode.NEAREST)
            if thickness is not None:
                thickness = TF.rotate(thickness, angle, interpolation=T.InterpolationMode.NEAREST)
            if orientation is not None:
                orientation = TF.rotate(orientation, angle, interpolation=T.InterpolationMode.NEAREST)
        
        # Random scale (0.9 to 1.1)
        if torch.rand(1).item() > 0.5:
            scale = 0.9 + torch.rand(1).item() * 0.2
            new_h = int(self.image_size[0] * scale)
            new_w = int(self.image_size[1] * scale)
            
            # Scale
            image = TF.resize(image, (new_h, new_w), interpolation=T.InterpolationMode.BILINEAR)
            vessel = TF.resize(vessel, (new_h, new_w), interpolation=T.InterpolationMode.NEAREST)
            mask = TF.resize(mask, (new_h, new_w), interpolation=T.InterpolationMode.NEAREST)
            if av is not None:
                av = TF.resize(av, (new_h, new_w), interpolation=T.InterpolationMode.NEAREST)
            if od is not None:
                od = TF.resize(od, (new_h, new_w), interpolation=T.InterpolationMode.NEAREST)
            if thickness is not None:
                thickness = TF.resize(thickness, (new_h, new_w), interpolation=T.InterpolationMode.NEAREST)
            if orientation is not None:
                orientation = TF.resize(orientation, (new_h, new_w), interpolation=T.InterpolationMode.NEAREST)
            
            # Center crop or pad back to original size
            if scale > 1:
                # Crop
                top = (new_h - self.image_size[0]) // 2
                left = (new_w - self.image_size[1]) // 2
                image = TF.crop(image, top, left, self.image_size[0], self.image_size[1])
                vessel = TF.crop(vessel, top, left, self.image_size[0], self.image_size[1])
                mask = TF.crop(mask, top, left, self.image_size[0], self.image_size[1])
                if av is not None:
                    av = TF.crop(av, top, left, self.image_size[0], self.image_size[1])
                if od is not None:
                    od = TF.crop(od, top, left, self.image_size[0], self.image_size[1])
                if thickness is not None:
                    thickness = TF.crop(thickness, top, left, self.image_size[0], self.image_size[1])
                if orientation is not None:
                    orientation = TF.crop(orientation, top, left, self.image_size[0], self.image_size[1])
            else:
                # Pad
                pad_h = self.image_size[0] - new_h
                pad_w = self.image_size[1] - new_w
                padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
                image = TF.pad(image, padding)
                vessel = TF.pad(vessel, padding)
                mask = TF.pad(mask, padding)
                if av is not None:
                    av = TF.pad(av, padding)
                if od is not None:
                    od = TF.pad(od, padding)
                if thickness is not None:
                    thickness = TF.pad(thickness, padding)
                if orientation is not None:
                    orientation = TF.pad(orientation, padding)
        
        return image, vessel, mask, av, od, thickness, orientation
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        
        # Load images
        image = self._load_image(self.image_paths[idx], mode='RGB')
        vessel = self._load_image(self.vessel_paths[idx], mode='L')
        
        # Load FOV mask
        if len(self.mask_paths) > 0:
            mask = self._load_image(self.mask_paths[idx], mode='L')
        else:
            # Create a default mask (all ones)
            mask = Image.new('L', image.size, 255)
        
        # Load optional AV labels (RGB color-coded)
        av = None
        av_rgb = None
        if self.av_paths is not None:
            av_rgb = self._load_image(self.av_paths[idx], mode='RGB')
            
        # Load optional OD labels
        od = None
        if self.od_paths is not None:
            od = self._load_image(self.od_paths[idx], mode='L')
        
        # Load optional thickness/orientation maps (paths pre-resolved at init)
        thickness = None
        orientation = None

        thick_path = self._thickness_paths[idx]
        if thick_path is not None:
            thickness = torch.from_numpy(np.load(thick_path)).float().unsqueeze(0)

        ori_path = self._orientation_paths[idx]
        if ori_path is not None:
            orientation = torch.from_numpy(np.load(ori_path)).float().unsqueeze(0)
        
        # Convert to tensors
        image = self.to_tensor(image)
        vessel = self.to_tensor(vessel)
        mask = self.to_tensor(mask)
        
        if av is not None:
            av = self.to_tensor(av)
        if od is not None:
            od = self.to_tensor(od)
        
        # Resize
        image = self.resize(image)
        vessel = self.resize_mask(vessel)
        mask = self.resize_mask(mask)
        
        if av is not None:
            av = self.resize_mask(av)
        if od is not None:
            od = self.resize_mask(od)
        if thickness is not None:
            thickness = self.resize_mask(thickness)
        if orientation is not None:
            orientation = self.resize_mask(orientation)
        
        # Apply augmentation if enabled
        if self.augment:
            image, vessel, mask, av, od, thickness, orientation = self._apply_augmentation(
                image, vessel, mask, av, od, thickness, orientation
            )
        
        # Normalize image
        image = self.normalize(image)
        
        # Binarize masks (convert to 0/1)
        vessel = (vessel > 0.5).float()
        mask = (mask > 0.5).float()
        
        # Prepare output
        sample = {
            'image': image,           # [3, H, W]
            'vessel': vessel,         # [1, H, W], binary
            'mask': mask,             # [1, H, W], binary
            'image_path': self.image_paths[idx]
        }
        
        if av_rgb is not None:
            # Convert RGB AV image to class indices
            # Red (255,0,0) = Artery = 1
            # Blue (0,0,255) = Vein = 2
            # Black (0,0,0) = Background = 0
            av_arr = np.array(av_rgb)
            av_labels = np.zeros(av_arr.shape[:2], dtype=np.int64)
            
            # Artery: Red channel high, others low
            artery_mask = (av_arr[:,:,0] > 200) & (av_arr[:,:,1] < 100) & (av_arr[:,:,2] < 100)
            av_labels[artery_mask] = 1
            
            # Vein: Blue channel high, others low
            vein_mask = (av_arr[:,:,2] > 200) & (av_arr[:,:,0] < 100) & (av_arr[:,:,1] < 100)
            av_labels[vein_mask] = 2
            
            # Convert to tensor and resize
            av = torch.from_numpy(av_labels).unsqueeze(0).float()
            av = self.resize_mask(av)
            
            # Apply augmentation if enabled (av was already handled above)
            sample['av'] = av.squeeze(0).long()  # [H, W], class indices
            
        if od is not None:
            sample['od'] = (od > 0.5).float()  # [1, H, W], binary
            
        if thickness is not None:
            sample['thickness'] = thickness  # [1, H, W]
            
        if orientation is not None:
            sample['orientation'] = orientation  # [1, H, W]
        
        return sample


class ConnectivityDataset(Dataset):
    """
    Dataset for training the Connectivity Classification Network.
    
    This dataset generates pairs of vessel segments for connectivity prediction.
    
    Args:
        base_dataset: RetinalDataset instance
        segment_pairs_dir: Directory containing precomputed segment pairs
    """
    
    def __init__(
        self,
        base_dataset: RetinalDataset,
        segment_pairs_dir: Optional[str] = None,
        max_pairs_per_image: int = 1000
    ):
        super().__init__()
        
        self.base_dataset = base_dataset
        self.segment_pairs_dir = segment_pairs_dir
        self.max_pairs_per_image = max_pairs_per_image
        
        # If precomputed pairs exist, load them
        self.pairs = []
        if segment_pairs_dir is not None and os.path.exists(segment_pairs_dir):
            self._load_precomputed_pairs()
    
    def _load_precomputed_pairs(self):
        """Load precomputed segment pairs from disk."""
        pair_files = glob.glob(os.path.join(self.segment_pairs_dir, '*.npz'))
        for pf in sorted(pair_files):
            data = np.load(pf, allow_pickle=True)
            self.pairs.extend(data['pairs'].tolist())
        print(f"Loaded {len(self.pairs)} precomputed segment pairs")
    
    def __len__(self) -> int:
        if len(self.pairs) > 0:
            return len(self.pairs)
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a pair of vessel segments for connectivity prediction.
        
        Returns:
            Dictionary containing:
                - 'features_i': Features for segment i
                - 'features_j': Features for segment j
                - 'coords_i': Coordinates of segment i pixel
                - 'coords_j': Coordinates of segment j pixel
                - 'connected': 1 if connected, 0 otherwise
        """
        if len(self.pairs) > 0:
            pair = self.pairs[idx]
            return {
                'image_idx': pair['image_idx'],
                'coords_i': torch.tensor(pair['coords_i']),
                'coords_j': torch.tensor(pair['coords_j']),
                'connected': torch.tensor(pair['connected'], dtype=torch.float32)
            }
        else:
            # Return base sample for feature extraction
            return self.base_dataset[idx]


class PairwiseFeatureDataset(Dataset):
    """
    Dataset that provides pairwise features for connectivity classification.
    
    This is used during training when we have precomputed feature maps and
    need to extract pairs of pixels for the connectivity network.
    
    Reference: Section 2.1.4 - "The connectivity classification is implemented 
    using fully connected (fc) layers on top of the concatenation of the 
    penultimate layer features"
    """
    
    def __init__(
        self,
        thickness_features: torch.Tensor,  # [N, C, H, W]
        orientation_features: torch.Tensor,  # [N, C, H, W]
        segment_masks: List[np.ndarray],  # List of segment label maps
        junction_pairs: List[List[Tuple[int, int, int]]],  # pairs within distance
        connectivity_labels: List[np.ndarray],  # GT connectivity
        distance_threshold: int = 10
    ):
        super().__init__()
        
        self.thickness_features = thickness_features
        self.orientation_features = orientation_features
        self.segment_masks = segment_masks
        self.junction_pairs = junction_pairs
        self.connectivity_labels = connectivity_labels
        self.distance_threshold = distance_threshold
        
        # Flatten pairs for indexing
        self.all_pairs = []
        for img_idx, pairs in enumerate(junction_pairs):
            for seg_i, seg_j, connected in pairs:
                self.all_pairs.append((img_idx, seg_i, seg_j, connected))
    
    def __len__(self) -> int:
        return len(self.all_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_idx, seg_i, seg_j, connected = self.all_pairs[idx]
        
        # Get feature maps for this image
        thick_feat = self.thickness_features[img_idx]  # [C, H, W]
        ori_feat = self.orientation_features[img_idx]  # [C, H, W]
        
        # Concatenate features
        features = torch.cat([thick_feat, ori_feat], dim=0)  # [2C, H, W]
        
        # Get segment masks
        seg_mask = self.segment_masks[img_idx]
        
        # Get representative pixels for each segment (centerline pixels)
        seg_i_coords = np.where(seg_mask == seg_i)
        seg_j_coords = np.where(seg_mask == seg_j)
        
        # Sample a pixel from each segment
        if len(seg_i_coords[0]) > 0:
            pix_i = (seg_i_coords[0][len(seg_i_coords[0])//2], 
                     seg_i_coords[1][len(seg_i_coords[1])//2])
        else:
            pix_i = (0, 0)
            
        if len(seg_j_coords[0]) > 0:
            pix_j = (seg_j_coords[0][len(seg_j_coords[0])//2], 
                     seg_j_coords[1][len(seg_j_coords[1])//2])
        else:
            pix_j = (0, 0)
        
        return {
            'features': features,
            'coord_i': torch.tensor(pix_i),
            'coord_j': torch.tensor(pix_j),
            'connected': torch.tensor(connected, dtype=torch.float32)
        }


class CombinedDataset(Dataset):
    """
    Combined dataset that merges multiple RetinalDataset instances.
    
    This allows training on both DRIVE and IOSTAR datasets together.
    
    Args:
        datasets: List of RetinalDataset instances to combine
    """
    
    def __init__(self, datasets: List[Dataset]):
        super().__init__()
        self.datasets = datasets
        
        # Build cumulative lengths for indexing
        self.cumulative_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_lengths.append(total)
        
        self.total_length = total
        print(f"CombinedDataset: {self.total_length} samples from {len(datasets)} datasets")
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample, routing to the appropriate dataset."""
        if idx < 0:
            idx = self.total_length + idx
        
        # Find which dataset this index belongs to
        for i, cumlen in enumerate(self.cumulative_lengths):
            if idx < cumlen:
                if i == 0:
                    local_idx = idx
                else:
                    local_idx = idx - self.cumulative_lengths[i - 1]
                return self.datasets[i][local_idx]
        
        raise IndexError(f"Index {idx} out of range for CombinedDataset of length {self.total_length}")
