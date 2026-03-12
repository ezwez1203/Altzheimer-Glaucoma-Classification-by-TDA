"""
Configuration file for Topology-Aware Retinal Artery-Vein Classification.

Based on: "Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"
Reference: Section 3.2 (Evaluation Details)
"""

import os


class Config:
    """Configuration class for training and evaluation."""
    
    # ============ Paths ============
    # Project root directory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # Data directories
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
    DRIVE_ROOT = os.path.join(DATA_ROOT, 'DRIVE')
    IOSTAR_ROOT = os.path.join(DATA_ROOT, 'IOSTAR')
    
    # Processed data directories
    PROCESSED_DATA_ROOT = os.path.join(PROJECT_ROOT, 'processed_data')
    THICKNESS_MAPS_DIR = os.path.join(PROCESSED_DATA_ROOT, 'thickness_maps')
    ORIENTATION_MAPS_DIR = os.path.join(PROCESSED_DATA_ROOT, 'orientation_maps')
    
    # Checkpoints and results
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    
    # ============ Model Architecture (Section 2.1.4) ============
    # Base network: VGG-16
    # Multi-scale features from concatenated VGG-16 features
    BACKBONE = 'vgg16'
    
    # Encoder output channels (VGG16 features)
    ENCODER_CHANNELS = [64, 128, 256, 512, 512]
    
    # Number of channels for multi-scale features (reduced for thickness/orientation)
    # Paper: "changed the number of channels of multi-scale features from 16 to 32"
    MULTISCALE_CHANNELS = 32
    
    # FC layers for connectivity classification (Section 2.1.4)
    # "the number of units of the first fc layer was set to 256"
    FC_UNITS = 256
    
    # ============ Input/Output Dimensions ============
    IMAGE_SIZE = (384, 384)  # Reduced from 512 for 8GB GPU  # Input image size
    NUM_VESSEL_CLASSES = 2   # Vessel vs Background
    NUM_AV_CLASSES = 3       # Artery, Vein, Background
    
    # Thickness quantization (Section 3.2)
    # "five classes using quantization boundaries of [1.5, 3, 5, 7]"
    THICKNESS_BOUNDARIES = [1.5, 3.0, 5.0, 7.0]
    NUM_THICKNESS_CLASSES = 5
    
    # Orientation quantization (Section 3.2)
    # "seven classes, including six classes evenly dividing the range of [-π/2, π/2] and one background class"
    NUM_ORIENTATION_CLASSES = 7
    
    # ============ Training Hyperparameters (Section 3.2) ============
    # Optimizer: Adam
    OPTIMIZER = 'adam'
    
    # Learning rates
    # Multi-task network: "used 1, 10, and 10^-3 for λsegm, λAV on DRIVE"
    # "0.01, 1, and 10^-2 on the IOSTAR set"
    # Connectivity network: "used 100/10 for λaux and 10^3/10^4"
    
    # Learning rate for DRIVE dataset
    LEARNING_RATE_DRIVE = {
        'od_segmentation': 1e-5,
        'multitask': 1e-3,
        'connectivity': 1e-3
    }
    
    # Learning rate for IOSTAR dataset
    LEARNING_RATE_IOSTAR = {
        'od_segmentation': 1e-5,
        'multitask': 1e-2,
        'connectivity': 1e-4
    }
    
    # Weight decay
    WEIGHT_DECAY = 0.0
    
    # ============ Loss Weights (Equations 2, 3) ============
    # Multi-task loss: L_segm_av = λsegm * L_vsegm + λAV * L_AV (Eq. 2)
    # DRIVE weights
    LAMBDA_SEGM_DRIVE = 1.0
    LAMBDA_AV_DRIVE = 10.0
    
    # IOSTAR weights
    LAMBDA_SEGM_IOSTAR = 0.01
    LAMBDA_AV_IOSTAR = 1.0
    
    # Connectivity loss weights (Eq. 3)
    # L_thick_ori_conn = λthick * L_thick + λori * L_ori + λconn * L_conn
    # "λaux = 0 in the first step. In the second step, we used 100/10 for λaux"
    LAMBDA_THICK = 100.0
    LAMBDA_ORI = 100.0
    LAMBDA_CONN = 1.0
    
    # ============ Training Settings ============
    BATCH_SIZE = 1  # Reduced for 8GB GPU (RTX 4060)
    NUM_EPOCHS_STEP1 = 100  # Step 1: Auxiliary tasks (OD, Vessel, AV)
    NUM_EPOCHS_STEP2 = 100  # Step 2: Connectivity network
    
    # Validation split
    VAL_SPLIT = 0.1
    
    # Data augmentation (Table 1)
    # "Horizontal flip, rotation (rot.), scaling, elastic deformation, cropping,
    #  random brightness, and contrast adjustment"
    AUGMENTATION = {
        'horizontal_flip': True,
        'rotation': True,
        'scale': True,
        'elastic_deform': False,  # N for connectivity on DRIVE
        'crop': False,  # N for connectivity
        'brightness': False,  # N for connectivity
        'contrast': False  # N for connectivity
    }
    
    # ============ Connectivity Classification ============
    # Threshold for connectivity probability (Section 3.2)
    # "threshold of 0.5 for the connectivity probability"
    CONNECTIVITY_THRESHOLD = 0.5
    
    # For tree-wise AV classification (Section 2.4)
    # "If the segment-wise probability of being an artery/vein is higher than 0.8"
    AV_REASSIGN_THRESHOLD = 0.8
    
    # ============ Random Seed ============
    RANDOM_SEED = 42
    
    # ============ Device ============
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    # ============ Pretrained Weights ============
    USE_PRETRAINED_VGG = True

    # ============ Backbone Selection ============
    USE_ATTENTION_UNET = False  # False = VGG16 backbone, True = Attention U-Net++


# Default configuration instance
config = Config()
