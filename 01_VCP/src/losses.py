"""
Loss functions for Topology-Aware Retinal Artery-Vein Classification.

Based on: "Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"
Reference: Section 2.1.4 and Equations 1, 2, 3

Loss Functions:
1. Pixelwise softmax cross-entropy loss (Eq. 1) - for all segmentation/classification tasks
2. Multi-task loss (Eq. 2) - weighted sum of vessel segmentation and AV classification losses
3. Connectivity loss (Eq. 3) - weighted sum of auxiliary losses and connectivity loss
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Commonly used as a complement to cross-entropy for medical image segmentation.
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [B, 1, H, W]
            targets: Binary ground truth [B, 1, H, W]
            mask: Optional FOV mask [B, 1, H, W]
            
        Returns:
            Dice loss (1 - Dice coefficient)
        """
        probs = torch.sigmoid(inputs)
        
        if mask is not None:
            probs = probs * mask
            targets = targets * mask
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Useful for vessel segmentation where background dominates.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [B, C, H, W] or [B, 1, H, W]
            targets: Ground truth [B, H, W] or [B, 1, H, W]
            mask: Optional FOV mask
            
        Returns:
            Focal loss
        """
        if inputs.shape[1] == 1:
            # Binary case
            probs = torch.sigmoid(inputs)
            targets = targets.float()
            
            ce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
            
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - p_t) ** self.gamma
            
            if self.alpha > 0:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                focal_weight = alpha_t * focal_weight
            
            loss = focal_weight * ce_loss
        else:
            # Multi-class case
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            probs = F.softmax(inputs, dim=1)
            
            # Get probability of true class
            targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
            p_t = (probs * targets_one_hot).sum(dim=1)
            
            focal_weight = (1 - p_t) ** self.gamma
            loss = focal_weight * ce_loss
        
        if mask is not None:
            loss = loss * mask.squeeze(1)
            return loss.sum() / (mask.sum() + 1e-6)
        
        return loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """
    Pixelwise Softmax Cross-Entropy Loss.
    
    Reference: Equation 1 - "For all five aforementioned tasks, we use pixelwise 
    softmax cross-entropy losses."
    
    L_cel(X) = -1/|X| * Σ O*(x_i) log O(x_i)
    
    where O*(x_i) and O(x_i) are the one-hot encoded GT label and the class prediction
    for pixel x_i, respectively.
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        centerline_only: bool = False
    ):
        """
        Args:
            class_weights: Optional class weights for handling imbalance
            ignore_index: Index to ignore in loss computation
            centerline_only: If True, only compute loss on centerline pixels
                           (for thickness/orientation as mentioned in paper)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.centerline_only = centerline_only
        # Register as buffer so it moves with .to(device) automatically
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        centerline_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [B, C, H, W]
            targets: Ground truth class indices [B, H, W]
            mask: Optional FOV mask [B, 1, H, W]
            centerline_mask: Optional centerline mask for thickness/orientation
            
        Returns:
            Cross-entropy loss
        """
        B, C, H, W = inputs.shape
        
        # Compute per-pixel loss
        loss = F.cross_entropy(
            inputs, targets, weight=self.class_weights,
            ignore_index=self.ignore_index, reduction='none'
        )  # [B, H, W]
        
        # Apply masks
        if self.centerline_only and centerline_mask is not None:
            # Reference: "We experimentally confine X to centerline pixels in 
            # thickness model training."
            loss = loss * centerline_mask.squeeze(1)
            num_pixels = centerline_mask.sum() + 1e-6
        elif mask is not None:
            loss = loss * mask.squeeze(1)
            num_pixels = mask.sum() + 1e-6
        else:
            num_pixels = B * H * W
        
        return loss.sum() / num_pixels


class BinaryCrossEntropyLoss(nn.Module):
    """Binary cross-entropy loss for vessel segmentation."""
    
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        else:
            self.pos_weight = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [B, 1, H, W]
            targets: Binary ground truth [B, 1, H, W]
            mask: Optional FOV mask [B, 1, H, W]

        Returns:
            BCE loss
        """
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task Loss for Joint Vessel Segmentation and AV Classification.
    
    Reference: Equation 2 - "The loss function for training this network is defined 
    as the weighted summation of two losses L_vsegm and L_AV..."
    
    L_segm_av = λ_segm * L_vsegm + λAV * L_AV
    
    Args:
        lambda_segm: Weight for vessel segmentation loss
        lambda_av: Weight for AV classification loss
    """
    
    def __init__(
        self,
        lambda_segm: float = 1.0,
        lambda_av: float = 10.0
    ):
        super().__init__()
        
        self.lambda_segm = lambda_segm
        self.lambda_av = lambda_av
        
        # Vessel segmentation uses binary cross-entropy with pos_weight
        # to handle class imbalance (vessels are ~5-10% of image)
        self.vessel_loss_fn = BinaryCrossEntropyLoss(pos_weight=10.0)
        
        # AV classification uses cross-entropy with class weights
        # Background dominates (~90%), so downweight it and upweight vessels
        # Class 0 = background, Class 1 = artery, Class 2 = vein
        av_class_weights = torch.tensor([0.1, 5.0, 5.0])
        self.av_loss_fn = WeightedCrossEntropyLoss(class_weights=av_class_weights)
    
    def forward(
        self,
        vessel_logits: torch.Tensor,
        av_logits: torch.Tensor,
        vessel_gt: torch.Tensor,
        av_gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vessel_logits: Vessel segmentation logits [B, 1, H, W]
            av_logits: AV classification logits [B, 3, H, W]
            vessel_gt: Vessel ground truth [B, 1, H, W]
            av_gt: AV ground truth [B, H, W]
            mask: FOV mask [B, 1, H, W]
            
        Returns:
            Dictionary with individual and total losses
        """
        # Compute individual losses
        loss_vessel = self.vessel_loss_fn(vessel_logits, vessel_gt, mask)
        loss_av = self.av_loss_fn(av_logits, av_gt, mask)
        
        # Weighted sum
        total_loss = self.lambda_segm * loss_vessel + self.lambda_av * loss_av
        
        return {
            'loss': total_loss,
            'loss_vessel': loss_vessel,
            'loss_av': loss_av
        }


class ConnectivityLoss(nn.Module):
    """
    Connectivity Classification Network Loss.
    
    Reference: Equation 3 - "The loss function for training the connectivity 
    classification network is then defined as the weighted summation..."
    
    L_thick_ori_conn = λ_thick * L_thick + λ_ori * L_ori + λ_conn * L_conn
    
    Training Strategy (Section 3.2):
    - Step 1 (λ_aux = 0): Train auxiliary tasks only (thickness, orientation)
    - Step 2: Train the whole network with all losses
    
    Args:
        lambda_thick: Weight for thickness classification loss
        lambda_ori: Weight for orientation classification loss  
        lambda_conn: Weight for connectivity classification loss
        num_thickness_classes: Number of thickness classes (default: 5)
        num_orientation_classes: Number of orientation classes (default: 7)
    """
    
    def __init__(
        self,
        lambda_thick: float = 100.0,
        lambda_ori: float = 100.0,
        lambda_conn: float = 1.0,
        num_thickness_classes: int = 5,
        num_orientation_classes: int = 7,
        centerline_only: bool = True
    ):
        super().__init__()
        
        self.lambda_thick = lambda_thick
        self.lambda_ori = lambda_ori
        self.lambda_conn = lambda_conn
        
        # Auxiliary losses (thickness and orientation)
        self.thickness_loss_fn = WeightedCrossEntropyLoss(
            centerline_only=centerline_only
        )
        self.orientation_loss_fn = WeightedCrossEntropyLoss(
            centerline_only=centerline_only
        )
        
        # Connectivity loss (binary classification)
        self.connectivity_loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        thickness_logits: Optional[torch.Tensor] = None,
        orientation_logits: Optional[torch.Tensor] = None,
        connectivity_logits: Optional[torch.Tensor] = None,
        thickness_gt: Optional[torch.Tensor] = None,
        orientation_gt: Optional[torch.Tensor] = None,
        connectivity_gt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        centerline_mask: Optional[torch.Tensor] = None,
        step: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            thickness_logits: Thickness classification logits [B, 5, H, W]
            orientation_logits: Orientation classification logits [B, 7, H, W]
            connectivity_logits: Connectivity prediction logits [B, 2] or [B, N, 2]
            thickness_gt: Thickness ground truth [B, H, W]
            orientation_gt: Orientation ground truth [B, H, W]
            connectivity_gt: Connectivity ground truth [B] or [B, N]
            mask: FOV mask
            centerline_mask: Centerline mask for thickness/orientation
            step: Training step (1 = auxiliary only, 2 = full)
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # Auxiliary losses (always computed)
        if thickness_logits is not None and thickness_gt is not None:
            loss_thick = self.thickness_loss_fn(
                thickness_logits, thickness_gt, mask, centerline_mask
            )
            losses['loss_thickness'] = loss_thick
            total_loss = total_loss + self.lambda_thick * loss_thick
        
        if orientation_logits is not None and orientation_gt is not None:
            loss_ori = self.orientation_loss_fn(
                orientation_logits, orientation_gt, mask, centerline_mask
            )
            losses['loss_orientation'] = loss_ori
            total_loss = total_loss + self.lambda_ori * loss_ori
        
        # Connectivity loss (only in step 2)
        if step == 2 and connectivity_logits is not None and connectivity_gt is not None:
            # Handle different shapes
            if connectivity_logits.dim() == 3:
                # [B, N, 2] -> reshape to [B*N, 2]
                B, N, _ = connectivity_logits.shape
                connectivity_logits = connectivity_logits.view(B * N, 2)
                connectivity_gt = connectivity_gt.view(B * N).long()
            
            loss_conn = self.connectivity_loss_fn(connectivity_logits, connectivity_gt)
            losses['loss_connectivity'] = loss_conn
            total_loss = total_loss + self.lambda_conn * loss_conn
        
        losses['loss'] = total_loss
        
        return losses


class OpticDiscLoss(nn.Module):
    """
    Loss for Optic Disc Segmentation.
    
    Reference: Equation 1 - Pixelwise softmax cross-entropy with optional Dice loss.
    
    "We used the learning rate value of 10^-5 for the optic disc segmentation network."
    """
    
    def __init__(self, use_dice: bool = True, dice_weight: float = 0.5):
        super().__init__()
        
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        
        self.bce_loss = BinaryCrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: OD segmentation logits [B, 1, H, W]
            targets: OD ground truth [B, 1, H, W]
            mask: FOV mask [B, 1, H, W]
            
        Returns:
            Dictionary with losses
        """
        loss_bce = self.bce_loss(logits, targets, mask)
        
        losses = {'loss_bce': loss_bce}
        
        if self.use_dice:
            loss_dice = self.dice_loss(logits, targets, mask)
            losses['loss_dice'] = loss_dice
            losses['loss'] = (1 - self.dice_weight) * loss_bce + self.dice_weight * loss_dice
        else:
            losses['loss'] = loss_bce
        
        return losses


class CombinedLoss(nn.Module):
    """
    Combined loss that can be used for end-to-end training of the full pipeline.
    
    Wraps all individual losses with configurable weights.
    """
    
    def __init__(
        self,
        # Multi-task weights
        lambda_vessel: float = 1.0,
        lambda_av: float = 10.0,
        # Connectivity weights
        lambda_thick: float = 100.0,
        lambda_ori: float = 100.0,
        lambda_conn: float = 1.0,
        # OD weight
        lambda_od: float = 1.0
    ):
        super().__init__()
        
        self.lambda_od = lambda_od
        
        self.od_loss = OpticDiscLoss()
        self.multitask_loss = MultiTaskLoss(lambda_vessel, lambda_av)
        self.connectivity_loss = ConnectivityLoss(lambda_thick, lambda_ori, lambda_conn)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        step: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for all tasks.
        
        Args:
            predictions: Dictionary containing model predictions
            targets: Dictionary containing ground truth
            step: Training step (1 or 2)
            
        Returns:
            Dictionary with all computed losses
        """
        all_losses = {}
        total_loss = 0.0
        
        # OD segmentation loss
        if 'od_logits' in predictions and 'od_gt' in targets:
            od_losses = self.od_loss(
                predictions['od_logits'],
                targets['od_gt'],
                targets.get('mask')
            )
            all_losses['od_loss'] = od_losses['loss']
            total_loss = total_loss + self.lambda_od * od_losses['loss']
        
        # Multi-task loss
        if 'vessel_logits' in predictions and 'vessel_gt' in targets:
            mt_losses = self.multitask_loss(
                predictions['vessel_logits'],
                predictions['av_logits'],
                targets['vessel_gt'],
                targets['av_gt'],
                targets.get('mask')
            )
            all_losses.update({f'mt_{k}': v for k, v in mt_losses.items()})
            total_loss = total_loss + mt_losses['loss']
        
        # Connectivity loss
        if 'thickness_logits' in predictions:
            conn_losses = self.connectivity_loss(
                predictions.get('thickness_logits'),
                predictions.get('orientation_logits'),
                predictions.get('connectivity_logits'),
                targets.get('thickness_gt'),
                targets.get('orientation_gt'),
                targets.get('connectivity_gt'),
                targets.get('mask'),
                targets.get('centerline_mask'),
                step=step
            )
            all_losses.update({f'conn_{k}': v for k, v in conn_losses.items()})
            total_loss = total_loss + conn_losses['loss']
        
        all_losses['total_loss'] = total_loss
        
        return all_losses
