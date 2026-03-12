"""
Training script for Topology-Aware Retinal Artery-Vein Classification.

Based on: "Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"
Reference: Section 3.2 (Evaluation Details)

Two-Step Training Strategy:
1. Step 1: Train auxiliary tasks (OD segmentation, Vessel segmentation, AV classification,
          Thickness/Orientation classification) with λ_aux = 0
2. Step 2: Train connectivity classification network with all losses

"We use a two-step training scheme. We firstly trained the network only using 
the auxiliary tasks, and then trained the whole network."
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.dataset import RetinalDataset, CombinedDataset
from src.models import (
    OpticDiscSegmentationNetwork,
    MultiTaskNetwork,
    FullConnectivityPipeline
)
from src.losses import (
    OpticDiscLoss,
    MultiTaskLoss,
    ConnectivityLoss
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> list:
    """
    Compute per-class F1 scores, skipping background class (0).

    Args:
        preds: flattened predictions [N]
        targets: flattened ground-truth labels [N]
        num_classes: total number of classes (including background)

    Returns:
        List of F1 scores for classes 1..num_classes-1
    """
    f1_scores = []
    for cls in range(1, num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().float()
        fp = ((preds == cls) & (targets != cls)).sum().float()
        fn = ((preds != cls) & (targets == cls)).sum().float()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        f1_scores.append(f1.item())
    return f1_scores


def open_log_file(config: Config, prefix: str, dataset_name: str):
    """
    Open a txt log file under results/logs/.

    Returns:
        (file handle, file path)
    """
    log_dir = os.path.join(config.RESULTS_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f'{prefix}_{dataset_name}_{timestamp}.txt')
    f = open(log_path, 'w')
    return f, log_path


def get_device(config: Config) -> torch.device:
    """Get the device for training."""
    if config.DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def create_dataloaders(
    config: Config,
    dataset_name: str = 'DRIVE',
    return_av: bool = True,
    return_od: bool = False,
    thickness_dir: Optional[str] = None,
    orientation_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration object
        dataset_name: 'DRIVE', 'IOSTAR', or 'COMBINED' (both datasets)
        return_av: Whether to return AV labels
        return_od: Whether to return OD labels
        thickness_dir: Directory with thickness maps
        orientation_dir: Directory with orientation maps
        
    Returns:
        train_loader, val_loader
    """
    if dataset_name == 'COMBINED':
        # Load both DRIVE and IOSTAR datasets
        drive_dataset = RetinalDataset(
            root_dir=config.DRIVE_ROOT,
            mode='train',
            image_size=config.IMAGE_SIZE,
            augment=True,
            return_av=return_av,
            return_od=return_od,
            thickness_dir=thickness_dir,
            orientation_dir=orientation_dir
        )
        iostar_dataset = RetinalDataset(
            root_dir=config.IOSTAR_ROOT,
            mode='train',
            image_size=config.IMAGE_SIZE,
            augment=True,
            return_av=return_av,
            return_od=return_od,
            thickness_dir=thickness_dir,
            orientation_dir=orientation_dir
        )
        train_dataset = CombinedDataset([drive_dataset, iostar_dataset])
    else:
        if dataset_name == 'DRIVE':
            root_dir = config.DRIVE_ROOT
        else:
            root_dir = config.IOSTAR_ROOT
        
        # Training dataset
        train_dataset = RetinalDataset(
            root_dir=root_dir,
            mode='train',
            image_size=config.IMAGE_SIZE,
            augment=True,
            return_av=return_av,
            return_od=return_od,
            thickness_dir=thickness_dir,
            orientation_dir=orientation_dir
        )
    
    # Split into train and validation
    val_size = int(len(train_dataset) * config.VAL_SPLIT)
    train_size = len(train_dataset) - val_size
    
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.RANDOM_SEED)
        )
    else:
        val_dataset = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        val_loader = None
    
    return train_loader, val_loader


def train_optic_disc_segmentation(
    config: Config,
    dataset_name: str = 'DRIVE',
    num_epochs: int = 100,
    checkpoint_path: Optional[str] = None
) -> OpticDiscSegmentationNetwork:
    """
    Train Optic Disc Segmentation Network (Figure 3a).
    
    Reference: "We used the learning rate value of 10^-5 for the optic disc 
    segmentation network."
    
    Args:
        config: Configuration object
        dataset_name: 'DRIVE' or 'IOSTAR'
        num_epochs: Number of training epochs
        checkpoint_path: Path to save/load checkpoint
        
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("Training Optic Disc Segmentation Network")
    print("="*60)
    
    device = get_device(config)
    set_seed(config.RANDOM_SEED)
    
    # Create model
    model = OpticDiscSegmentationNetwork(pretrained=config.USE_PRETRAINED_VGG)
    model = model.to(device)
    
    # Load checkpoint if exists
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {start_epoch}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config, dataset_name, return_av=False, return_od=True
    )
    
    # Loss and optimizer
    criterion = OpticDiscLoss()
    
    lr = config.LEARNING_RATE_DRIVE['od_segmentation'] if dataset_name == 'DRIVE' \
        else config.LEARNING_RATE_IOSTAR['od_segmentation']
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    
    # TensorBoard writer + txt log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.RESULTS_DIR, 'logs', f'od_segm_{dataset_name}_{timestamp}')
    writer = SummaryWriter(log_dir)
    log_file, log_path = open_log_file(config, 'od_segm', dataset_name)
    log_file.write("epoch\ttrain_loss\tval_loss\tval_f1_od\n")
    print(f"Logging to: {log_path}")

    # Training loop
    best_loss = float('inf')
    best_f1 = 0.0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            if 'od' in batch:
                od_gt = batch['od'].to(device)
            else:
                B, _, H, W = images.shape
                od_gt = torch.zeros(B, 1, H, W, device=device)

            optimizer.zero_grad()
            logits = model(images)
            losses = criterion(logits, od_gt, masks)
            loss = losses['loss']
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation
        val_loss = 0.0
        val_f1_od = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    if 'od' in batch:
                        od_gt = batch['od'].to(device)
                    else:
                        B, _, H, W = images.shape
                        od_gt = torch.zeros(B, 1, H, W, device=device)
                    logits = model(images)
                    losses = criterion(logits, od_gt, masks)
                    val_loss += losses['loss'].item()

                    # OD F1: binary (background=0, od=1)
                    preds = (torch.sigmoid(logits) > 0.5).long().view(-1)
                    gt_bin = (od_gt > 0.5).long().view(-1)
                    tp = ((preds == 1) & (gt_bin == 1)).sum().float()
                    fp = ((preds == 1) & (gt_bin == 0)).sum().float()
                    fn = ((preds == 0) & (gt_bin == 1)).sum().float()
                    val_f1_od += (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

            val_loss /= len(val_loader)
            val_f1_od /= len(val_loader)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('F1/od', val_f1_od, epoch)

        log_file.write(f"{epoch+1}\t{avg_train_loss:.6f}\t{val_loss:.6f}\t{val_f1_od:.6f}\n")
        log_file.flush()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, OD F1: {val_f1_od:.4f}")

        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            save_path = os.path.join(config.CHECKPOINT_DIR, f'od_segm_{dataset_name}_best.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, save_path)

    log_file.write(f"\nBest train loss: {best_loss:.6f}\n")
    log_file.close()
    writer.close()
    return model


def train_multitask_network(
    config: Config,
    dataset_name: str = 'DRIVE',
    num_epochs: int = 100,
    checkpoint_path: Optional[str] = None
) -> MultiTaskNetwork:
    """
    Train Multi-task Network for Vessel Segmentation and AV Classification (Figure 3b).
    
    Reference: Section 3.2 - "When training the multi-task network for binary vessel 
    segmentation and AV classification, we used 1, 10, and 10^-3 for λ_segm, λ_AV, 
    and the learning rate on the DRIVE set"
    
    Args:
        config: Configuration object
        dataset_name: 'DRIVE' or 'IOSTAR'
        num_epochs: Number of training epochs
        checkpoint_path: Path to save/load checkpoint
        
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("Training Multi-task Network (Vessel Segmentation + AV Classification)")
    print("="*60)
    
    device = get_device(config)
    set_seed(config.RANDOM_SEED)
    
    # Create model
    model = MultiTaskNetwork(
        pretrained=config.USE_PRETRAINED_VGG,
        num_av_classes=config.NUM_AV_CLASSES,
        multiscale_channels=config.MULTISCALE_CHANNELS,
        use_attention_unet=config.USE_ATTENTION_UNET
    )
    model = model.to(device)
    
    # Load checkpoint if exists
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {start_epoch}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config, dataset_name, return_av=True
    )
    
    # Loss function
    if dataset_name == 'DRIVE':
        lambda_segm = config.LAMBDA_SEGM_DRIVE
        lambda_av = config.LAMBDA_AV_DRIVE
        lr = config.LEARNING_RATE_DRIVE['multitask']
    else:
        lambda_segm = config.LAMBDA_SEGM_IOSTAR
        lambda_av = config.LAMBDA_AV_IOSTAR
        lr = config.LEARNING_RATE_IOSTAR['multitask']
    
    criterion = MultiTaskLoss(lambda_segm=lambda_segm, lambda_av=lambda_av)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # TensorBoard writer + txt log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.RESULTS_DIR, 'logs', f'multitask_{dataset_name}_{timestamp}')
    writer = SummaryWriter(log_dir)
    log_file, log_path = open_log_file(config, 'multitask', dataset_name)
    log_file.write("epoch\ttrain_loss\tval_loss\tf1_artery\tf1_vein\tf1_macro\n")
    print(f"Logging to: {log_path}")

    # Training loop
    best_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = {'loss': 0.0, 'loss_vessel': 0.0, 'loss_av': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            vessel_gt = batch['vessel'].to(device)
            masks = batch['mask'].to(device)

            if 'av' in batch:
                av_gt = batch['av'].to(device)
            else:
                av_gt = vessel_gt.squeeze(1).long()

            optimizer.zero_grad()
            vessel_logits, av_logits, _ = model(images)
            losses = criterion(vessel_logits, av_logits, vessel_gt, av_gt, masks)
            loss = losses['loss']
            loss.backward()
            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            pbar.set_postfix({'loss': loss.item()})

        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
            writer.add_scalar(f'Loss/{key}', epoch_losses[key], epoch)

        # Validation
        val_loss = 0.0
        val_f1_artery = 0.0
        val_f1_vein = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    vessel_gt = batch['vessel'].to(device)
                    masks = batch['mask'].to(device)
                    if 'av' in batch:
                        av_gt = batch['av'].to(device)
                    else:
                        av_gt = vessel_gt.squeeze(1).long()

                    vessel_logits, av_logits, _ = model(images)
                    losses = criterion(vessel_logits, av_logits, vessel_gt, av_gt, masks)
                    val_loss += losses['loss'].item()

                    # AV F1: classes 1=artery, 2=vein
                    av_preds = av_logits.argmax(dim=1).view(-1)
                    av_targets = av_gt.view(-1)
                    f1s = compute_f1(av_preds, av_targets, num_classes=config.NUM_AV_CLASSES)
                    val_f1_artery += f1s[0]
                    val_f1_vein += f1s[1] if len(f1s) > 1 else 0.0

            n_val = len(val_loader)
            val_loss /= n_val
            val_f1_artery /= n_val
            val_f1_vein /= n_val
            val_f1_macro = (val_f1_artery + val_f1_vein) / 2.0
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('F1/artery', val_f1_artery, epoch)
            writer.add_scalar('F1/vein', val_f1_vein, epoch)
            writer.add_scalar('F1/macro', val_f1_macro, epoch)
        else:
            val_f1_macro = 0.0

        log_file.write(
            f"{epoch+1}\t{epoch_losses['loss']:.6f}\t{val_loss:.6f}\t"
            f"{val_f1_artery:.6f}\t{val_f1_vein:.6f}\t{val_f1_macro:.6f}\n"
        )
        log_file.flush()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_losses['loss']:.4f}, "
              f"Vessel: {epoch_losses['loss_vessel']:.4f}, AV: {epoch_losses['loss_av']:.4f}, "
              f"F1 A/V/Macro: {val_f1_artery:.4f}/{val_f1_vein:.4f}/{val_f1_macro:.4f}")

        scheduler.step(epoch_losses['loss'])

        if epoch_losses['loss'] < best_loss:
            best_loss = epoch_losses['loss']
            save_path = os.path.join(config.CHECKPOINT_DIR, f'multitask_{dataset_name}_best.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, save_path)

    log_file.write(f"\nBest train loss: {best_loss:.6f}\n")
    log_file.close()
    writer.close()
    return model


def train_connectivity_network(
    config: Config,
    dataset_name: str = 'DRIVE',
    num_epochs_step1: int = 50,
    num_epochs_step2: int = 50,
    checkpoint_path: Optional[str] = None
) -> FullConnectivityPipeline:
    """
    Train Connectivity Classification Network with Two-Step Strategy (Figure 3c).
    
    Reference: Section 3.2 - "As for the training of the connectivity classification 
    network, we use a two-step training scheme. We firstly trained the network only 
    using the auxiliary tasks, and then trained the whole network. Thus, we set 
    λ_aux = 0 in the first step. In the second step, we used 100/10 for λ_aux..."
    
    Step 1: Train thickness/orientation classification (auxiliary tasks)
    Step 2: Train connectivity prediction with all losses
    
    Args:
        config: Configuration object
        dataset_name: 'DRIVE' or 'IOSTAR'
        num_epochs_step1: Epochs for Step 1 (auxiliary only)
        num_epochs_step2: Epochs for Step 2 (full training)
        checkpoint_path: Path to save/load checkpoint
        
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("Training Connectivity Classification Network (Two-Step)")
    print("="*60)
    
    device = get_device(config)
    set_seed(config.RANDOM_SEED)
    
    # Create model
    model = FullConnectivityPipeline(
        pretrained=config.USE_PRETRAINED_VGG,
        multiscale_channels=config.MULTISCALE_CHANNELS,
        num_thickness_classes=config.NUM_THICKNESS_CLASSES,
        num_orientation_classes=config.NUM_ORIENTATION_CLASSES,
        fc_units=config.FC_UNITS,
        use_attention_unet=config.USE_ATTENTION_UNET
    )
    model = model.to(device)
    
    # Create dataloaders with thickness/orientation maps
    # Files may be in original locations (DRIVE_training/) or organized locations (thickness_maps/)
    if dataset_name == 'COMBINED':
        # Check for organized paths first (thickness_maps/, orientation_maps/)
        thickness_maps_dir = os.path.join(config.PROCESSED_DATA_ROOT, 'thickness_maps')
        orientation_maps_dir = os.path.join(config.PROCESSED_DATA_ROOT, 'orientation_maps')
        
        # Also check original preprocessing output directories
        drive_processed_dir = os.path.join(config.PROCESSED_DATA_ROOT, 'DRIVE_training')
        iostar_processed_dir = os.path.join(config.PROCESSED_DATA_ROOT, 'IOSTAR_training')
        
        # Determine which directories to use
        if os.path.exists(thickness_maps_dir) and len(os.listdir(thickness_maps_dir)) > 0:
            # Use organized paths (all files in one place)
            print(f"Found organized data: {thickness_maps_dir} and {orientation_maps_dir}")
            thickness_dir = thickness_maps_dir
            orientation_dir = orientation_maps_dir
            
            # For combined dataset with organized paths, create datasets without separate dirs
            # The thickness/orientation lookup will match by image basename
            drive_dataset = RetinalDataset(
                root_dir=config.DRIVE_ROOT,
                mode='train',
                image_size=config.IMAGE_SIZE,
                augment=True,
                return_av=False,
                thickness_dir=thickness_dir,
                orientation_dir=orientation_dir
            )
            iostar_dataset = RetinalDataset(
                root_dir=config.IOSTAR_ROOT,
                mode='train',
                image_size=config.IMAGE_SIZE,
                augment=True,
                return_av=False,
                thickness_dir=thickness_dir,
                orientation_dir=orientation_dir
            )
            train_dataset = CombinedDataset([drive_dataset, iostar_dataset])
            
            # Split into train and validation
            val_size = int(len(train_dataset) * config.VAL_SPLIT)
            train_size = len(train_dataset) - val_size
            
            if val_size > 0:
                train_dataset, val_dataset = random_split(
                    train_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(config.RANDOM_SEED)
                )
            else:
                val_dataset = None
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE) if val_dataset else None
        else:
            print(f"Warning: No processed data found for COMBINED dataset")
            train_loader, val_loader = create_dataloaders(config, dataset_name, return_av=False)
    else:
        processed_subdir = f'{dataset_name}_training'
        processed_dir = os.path.join(config.PROCESSED_DATA_ROOT, processed_subdir)
        
        # Check if processed data exists
        if os.path.exists(processed_dir):
            thickness_dir = processed_dir
            orientation_dir = processed_dir
            print(f"Found processed data in: {processed_dir}")
        else:
            thickness_dir = None
            orientation_dir = None
            print(f"Warning: No processed data found in {processed_dir}")
        
        train_loader, val_loader = create_dataloaders(
            config, dataset_name,
            return_av=False,
            thickness_dir=thickness_dir,
            orientation_dir=orientation_dir
        )
    
    # Get learning rate
    if dataset_name == 'DRIVE':
        lr = config.LEARNING_RATE_DRIVE['connectivity']
    else:
        lr = config.LEARNING_RATE_IOSTAR['connectivity']
    
    # Loss function
    criterion = ConnectivityLoss(
        lambda_thick=config.LAMBDA_THICK,
        lambda_ori=config.LAMBDA_ORI,
        lambda_conn=config.LAMBDA_CONN,
        num_thickness_classes=config.NUM_THICKNESS_CLASSES,
        num_orientation_classes=config.NUM_ORIENTATION_CLASSES,
        centerline_only=True
    )
    
    # TensorBoard writer + txt log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.RESULTS_DIR, 'logs', f'connectivity_{dataset_name}_{timestamp}')
    writer = SummaryWriter(log_dir)
    log_file, log_path = open_log_file(config, 'connectivity', dataset_name)
    log_file.write("step\tepoch\ttrain_loss\n")
    print(f"Logging to: {log_path}")

    # ========== STEP 1: Train Auxiliary Tasks Only ==========
    print("\n--- Step 1: Training Auxiliary Tasks (Thickness/Orientation) ---")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    
    for epoch in range(num_epochs_step1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Step1 Epoch {epoch+1}/{num_epochs_step1}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            vessel_gt = batch['vessel'].to(device)
            
            # Get thickness/orientation GT if available
            if 'thickness' in batch and 'orientation' in batch:
                thickness_gt = batch['thickness'].squeeze(1).long().to(device)
                orientation_gt = batch['orientation'].squeeze(1).long().to(device)
            else:
                # Skip if no ground truth available
                continue
            
            # Create centerline mask from vessel GT
            centerline_mask = vessel_gt  # Simplified - use full vessel mask
            
            optimizer.zero_grad()
            
            # Forward pass (no connectivity)
            thick_logits, ori_logits, _ = model(images)
            
            # Compute loss (Step 1: auxiliary only, λ_conn = 0)
            losses = criterion(
                thickness_logits=thick_logits,
                orientation_logits=ori_logits,
                thickness_gt=thickness_gt,
                orientation_gt=orientation_gt,
                centerline_mask=centerline_mask,
                step=1  # Auxiliary only
            )
            
            loss = losses['loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / max(len(train_loader), 1)
        writer.add_scalar('Loss/step1', avg_loss, epoch)
        log_file.write(f"1\t{epoch+1}\t{avg_loss:.6f}\n")
        log_file.flush()

        print(f"Step1 Epoch {epoch+1}/{num_epochs_step1}, Loss: {avg_loss:.4f}")
    
    # Save Step 1 checkpoint
    step1_path = os.path.join(config.CHECKPOINT_DIR, f'connectivity_{dataset_name}_step1.pth')
    os.makedirs(os.path.dirname(step1_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': 1
    }, step1_path)
    
    # ========== STEP 2: Train Full Network ==========
    print("\n--- Step 2: Training Full Network (Including Connectivity) ---")
    
    # Reset optimizer for Step 2
    optimizer = optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=config.WEIGHT_DECAY)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs_step2):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Step2 Epoch {epoch+1}/{num_epochs_step2}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            vessel_gt = batch['vessel'].to(device)
            
            # Get GT if available
            if 'thickness' in batch and 'orientation' in batch:
                thickness_gt = batch['thickness'].squeeze(1).long().to(device)
                orientation_gt = batch['orientation'].squeeze(1).long().to(device)
            else:
                continue
            
            centerline_mask = vessel_gt
            
            optimizer.zero_grad()
            
            # For Step 2, we need connectivity pairs
            # For simplicity, we create random pairs within the vessel
            B, _, H, W = images.shape
            
            # Sample random coordinate pairs within vessels
            num_pairs = 64
            coords_i = torch.zeros(B, num_pairs, 2, device=device)
            coords_j = torch.zeros(B, num_pairs, 2, device=device)
            connectivity_gt = torch.zeros(B, num_pairs, device=device)
            
            for b in range(B):
                vessel_coords = torch.where(vessel_gt[b, 0] > 0)
                if len(vessel_coords[0]) > 1:
                    num_points = len(vessel_coords[0])
                    # Vectorized: sample all num_pairs indices at once
                    idx_i = torch.randint(0, num_points, (num_pairs,))
                    idx_j = torch.randint(0, num_points, (num_pairs,))
                    coords_i[b, :, 0] = vessel_coords[0][idx_i].float()
                    coords_i[b, :, 1] = vessel_coords[1][idx_i].float()
                    coords_j[b, :, 0] = vessel_coords[0][idx_j].float()
                    coords_j[b, :, 1] = vessel_coords[1][idx_j].float()
                    # Vectorized distance and connectivity label
                    dy = coords_i[b, :, 0] - coords_j[b, :, 0]
                    dx = coords_i[b, :, 1] - coords_j[b, :, 1]
                    connectivity_gt[b] = (dy * dy + dx * dx < 400).float()  # 20² = 400
            
            # Forward pass with connectivity
            thick_logits, ori_logits, conn_logits = model(
                images, coords_i, coords_j
            )
            
            # Compute loss (Step 2: all losses)
            losses = criterion(
                thickness_logits=thick_logits,
                orientation_logits=ori_logits,
                connectivity_logits=conn_logits,
                thickness_gt=thickness_gt,
                orientation_gt=orientation_gt,
                connectivity_gt=connectivity_gt,
                centerline_mask=centerline_mask,
                step=2  # Full training
            )
            
            loss = losses['loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / max(len(train_loader), 1)
        writer.add_scalar('Loss/step2', avg_loss, epoch)
        log_file.write(f"2\t{epoch+1}\t{avg_loss:.6f}\n")
        log_file.flush()

        print(f"Step2 Epoch {epoch+1}/{num_epochs_step2}, Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'connectivity_{dataset_name}_best.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'step': 2
            }, save_path)

    log_file.write(f"\nBest step2 loss: {best_loss:.6f}\n")
    log_file.close()
    writer.close()
    return model


def train_full_pipeline(
    config: Config,
    dataset_name: str = 'DRIVE'
):
    """
    Train the complete pipeline following the paper's approach.
    
    Training order:
    1. Optic Disc Segmentation Network
    2. Multi-task Network (Vessel + AV)
    3. Connectivity Network (Two-step)
    """
    print("\n" + "="*70)
    print("FULL TRAINING PIPELINE")
    print(f"Dataset: {dataset_name}")
    print("="*70)
    
    start_time = time.time()
    
    # Step 1: Train Optic Disc Segmentation
    # Note: OD training uses only DRIVE (IOSTAR doesn't have OD labels)
    print("\n[1/3] Optic Disc Segmentation Network")
    if dataset_name == 'COMBINED':
        print("  Using DRIVE only for OD segmentation (IOSTAR lacks OD labels)")
        od_model = train_optic_disc_segmentation(
            config, 'DRIVE',
            num_epochs=config.NUM_EPOCHS_STEP1 // 2
        )
    else:
        od_model = train_optic_disc_segmentation(
            config, dataset_name,
            num_epochs=config.NUM_EPOCHS_STEP1 // 2
        )
    
    # Step 2: Train Multi-task Network
    print("\n[2/3] Multi-task Network")
    mt_model = train_multitask_network(
        config, dataset_name,
        num_epochs=config.NUM_EPOCHS_STEP1
    )
    
    # Step 3: Train Connectivity Network
    print("\n[3/3] Connectivity Classification Network")
    conn_model = train_connectivity_network(
        config, dataset_name,
        num_epochs_step1=config.NUM_EPOCHS_STEP1 // 2,
        num_epochs_step2=config.NUM_EPOCHS_STEP2
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/3600:.2f} hours")
    
    return od_model, mt_model, conn_model


def main():
    parser = argparse.ArgumentParser(
        description='Train Topology-Aware Retinal AV Classification Networks'
    )
    parser.add_argument('--dataset', type=str, default='COMBINED',
                        choices=['DRIVE', 'IOSTAR', 'COMBINED'],
                        help='Dataset to train on (COMBINED uses both DRIVE + IOSTAR)')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'od', 'multitask', 'connectivity'],
                        help='Which network to train')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    num_epochs = args.epochs or config.NUM_EPOCHS_STEP1
    
    # Create output directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Train
    if args.task == 'all':
        train_full_pipeline(config, args.dataset)
    elif args.task == 'od':
        train_optic_disc_segmentation(
            config, args.dataset, num_epochs, args.checkpoint
        )
    elif args.task == 'multitask':
        train_multitask_network(
            config, args.dataset, num_epochs, args.checkpoint
        )
    elif args.task == 'connectivity':
        train_connectivity_network(
            config, args.dataset,
            num_epochs_step1=num_epochs // 2,
            num_epochs_step2=num_epochs // 2,
            checkpoint_path=args.checkpoint
        )


if __name__ == '__main__':
    main()
