"""
Neural Network Models for Topology-Aware Retinal Artery-Vein Classification.

Based on: "Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"
Reference: Section 2.1.4 (Network Architecture) and Figure 3

Three networks are implemented:
(a) Optic Disc Segmentation Network - VGG16 Encoder + Simple Decoder
(b) Multi-task Network - Shared Encoder + Two Decoders (Vessel Seg, AV Classification)
(c) Connectivity Network - Accepts features from Thickness/Orientation maps with Hadamard product
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =============================================================================
# Attention U-Net++ Components (Enhanced VCP Backbone)
# Reference: Screenshot - "Backbone: Attention U-Net++"
# =============================================================================

class ConvBlock(nn.Module):
    """Standard convolutional block with BatchNorm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batchnorm: bool = True
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=not use_batchnorm),
        ])
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate for Attention U-Net++.
    
    Reference: "Attention U-Net: Learning Where to Look for the Pancreas"
    (Oktay et al., 2018)
    
    The attention gate learns to focus on relevant features by computing
    attention coefficients that weight the skip connection features.
    
    Args:
        F_g: Number of feature channels in gating signal
        F_l: Number of feature channels in low-level features
        F_int: Number of intermediate channels
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        # Gating signal transformation
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Low-level feature transformation
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Attention coefficient computation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder (coarser resolution) [B, F_g, H, W]
            x: Low-level features from encoder (skip connection) [B, F_l, H, W]
            
        Returns:
            Attention-weighted features [B, F_l, H, W]
        """
        # Transform gating signal
        g1 = self.W_g(g)
        
        # Transform low-level features
        x1 = self.W_x(x)
        
        # Handle size mismatch
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        # Compute attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention
        return x * psi


class AttentionUNetPlusPlusEncoder(nn.Module):
    """
    Attention U-Net++ Encoder.
    
    Enhanced U-Net++ with Attention Gates for the VCP model.
    Reference: Screenshot - "Step 3 Enhanced VCP (Core) 혈관 분할 + 동/정맥 분류 + 연결성 예측 Backbone: Attention U-Net++"
    
    This encoder combines:
    1. U-Net++ nested dense skip connections
    2. Attention Gates for focusing on relevant features
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        base_channels: Base number of channels (default: 64)
        deep_supervision: Whether to use deep supervision outputs
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        deep_supervision: bool = False
    ):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        
        # Channel configuration: [64, 128, 256, 512, 1024]
        nb_filter = [base_channels, base_channels * 2, base_channels * 4, 
                     base_channels * 8, base_channels * 16]
        
        self.out_channels = nb_filter[:5]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Encoder path (x_0,0 -> x_1,0 -> x_2,0 -> x_3,0 -> x_4,0)
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])
        
        # Attention Gates for each decoder level
        # Level 1 attention gates
        self.att0_1 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0] // 2)
        
        # Level 2 attention gates
        self.att1_1 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1] // 2)
        self.att0_2 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0] // 2)
        
        # Level 3 attention gates
        self.att2_1 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2] // 2)
        self.att1_2 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1] // 2)
        self.att0_3 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0] // 2)
        
        # Level 4 attention gates
        self.att3_1 = AttentionGate(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[3] // 2)
        self.att2_2 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2] // 2)
        self.att1_3 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1] // 2)
        self.att0_4 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0] // 2)
        
        # Nested decoder path (U-Net++ dense connections)
        # Column 1
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        
        # Column 2
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        
        # Column 3
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        
        # Column 4
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        
        # Deep supervision outputs (optional)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            List of feature tensors at each scale for compatibility with VGG16Encoder
            Returns features from: [x0_4, x1_3, x2_2, x3_1, x4_0] (finest to coarsest)
        """
        # Encoder path
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        # Nested decoder path with attention gates
        # Column 1
        x0_0_att = self.att0_1(g=self.up(x1_0), x=x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0_att, self.up(x1_0)], dim=1))
        
        # Column 2
        x1_0_att = self.att1_1(g=self.up(x2_0), x=x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0_att, self.up(x2_0)], dim=1))
        
        x0_0_att2 = self.att0_2(g=self.up(x1_1), x=x0_0)
        x0_1_att2 = self.att0_2(g=self.up(x1_1), x=x0_1)
        x0_2 = self.conv0_2(torch.cat([x0_0_att2, x0_1_att2, self.up(x1_1)], dim=1))
        
        # Column 3
        x2_0_att = self.att2_1(g=self.up(x3_0), x=x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0_att, self.up(x3_0)], dim=1))
        
        x1_0_att2 = self.att1_2(g=self.up(x2_1), x=x1_0)
        x1_1_att2 = self.att1_2(g=self.up(x2_1), x=x1_1)
        x1_2 = self.conv1_2(torch.cat([x1_0_att2, x1_1_att2, self.up(x2_1)], dim=1))
        
        x0_0_att3 = self.att0_3(g=self.up(x1_2), x=x0_0)
        x0_1_att3 = self.att0_3(g=self.up(x1_2), x=x0_1)
        x0_2_att3 = self.att0_3(g=self.up(x1_2), x=x0_2)
        x0_3 = self.conv0_3(torch.cat([x0_0_att3, x0_1_att3, x0_2_att3, self.up(x1_2)], dim=1))
        
        # Column 4
        x3_0_att = self.att3_1(g=self.up(x4_0), x=x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0_att, self.up(x4_0)], dim=1))
        
        x2_0_att2 = self.att2_2(g=self.up(x3_1), x=x2_0)
        x2_1_att2 = self.att2_2(g=self.up(x3_1), x=x2_1)
        x2_2 = self.conv2_2(torch.cat([x2_0_att2, x2_1_att2, self.up(x3_1)], dim=1))
        
        x1_0_att3 = self.att1_3(g=self.up(x2_2), x=x1_0)
        x1_1_att3 = self.att1_3(g=self.up(x2_2), x=x1_1)
        x1_2_att3 = self.att1_3(g=self.up(x2_2), x=x1_2)
        x1_3 = self.conv1_3(torch.cat([x1_0_att3, x1_1_att3, x1_2_att3, self.up(x2_2)], dim=1))
        
        x0_0_att4 = self.att0_4(g=self.up(x1_3), x=x0_0)
        x0_1_att4 = self.att0_4(g=self.up(x1_3), x=x0_1)
        x0_2_att4 = self.att0_4(g=self.up(x1_3), x=x0_2)
        x0_3_att4 = self.att0_4(g=self.up(x1_3), x=x0_3)
        x0_4 = self.conv0_4(torch.cat([x0_0_att4, x0_1_att4, x0_2_att4, x0_3_att4, self.up(x1_3)], dim=1))
        
        # Return multi-scale features for compatibility with existing decoders
        # Mapped to match VGG16Encoder output format: [f1, f2, f3, f4, f5]
        # where f1 is finest resolution and f5 is coarsest
        features = [x0_4, x1_3, x2_2, x3_1, x4_0]
        
        return features
    
    def get_deep_supervision_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get deep supervision outputs for training."""
        features = self.forward(x)
        
        if self.deep_supervision:
            outputs = [
                self.final1(features[0]),  # x0_4
                self.final2(self.up(features[1])),  # x0_3 approximation
                self.final3(self.up(self.up(features[2]))),  # x0_2 approximation
                self.final4(self.up(self.up(self.up(features[3]))))  # x0_1 approximation
            ]
            return outputs
        
        return features


class VGG16Encoder(nn.Module):
    """
    VGG-16 based encoder for feature extraction.
    
    Reference: Section 2.1.4 - "We adopt the network in [7], which is based on 
    the VGG-16 network [15], as our base network."
    
    The encoder extracts multi-scale features that are later concatenated
    and resized to have identical spatial resolutions.
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained VGG16
        if pretrained:
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg16 = models.vgg16(weights=None)
        
        features = list(vgg16.features.children())
        
        # Split VGG16 into encoder stages
        # Stage 1: conv1_1, conv1_2, pool1 -> 64 channels
        self.stage1 = nn.Sequential(*features[0:5])
        # Stage 2: conv2_1, conv2_2, pool2 -> 128 channels
        self.stage2 = nn.Sequential(*features[5:10])
        # Stage 3: conv3_1, conv3_2, conv3_3, pool3 -> 256 channels
        self.stage3 = nn.Sequential(*features[10:17])
        # Stage 4: conv4_1, conv4_2, conv4_3, pool4 -> 512 channels
        self.stage4 = nn.Sequential(*features[17:24])
        # Stage 5: conv5_1, conv5_2, conv5_3, pool5 -> 512 channels
        self.stage5 = nn.Sequential(*features[24:31])
        
        # Output channels at each stage
        self.out_channels = [64, 128, 256, 512, 512]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            List of feature tensors at each scale
        """
        features = []
        
        f1 = self.stage1(x)    # [B, 64, H/2, W/2]
        features.append(f1)
        
        f2 = self.stage2(f1)   # [B, 128, H/4, W/4]
        features.append(f2)
        
        f3 = self.stage3(f2)   # [B, 256, H/8, W/8]
        features.append(f3)
        
        f4 = self.stage4(f3)   # [B, 512, H/16, W/16]
        features.append(f4)
        
        f5 = self.stage5(f4)   # [B, 512, H/32, W/32]
        features.append(f5)
        
        return features


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SimpleDecoder(nn.Module):
    """
    Simple decoder for segmentation tasks (OD segmentation).
    
    Reference: Figure 3(a) - "Network for optic disc segmentation (OD segm.)"
    """
    
    def __init__(
        self,
        encoder_channels: List[int] = [64, 128, 256, 512, 512],
        num_classes: int = 1
    ):
        super().__init__()
        
        self.encoder_channels = encoder_channels
        
        # Decoder stages
        self.decoder5 = DecoderBlock(encoder_channels[4], encoder_channels[3], 256)
        self.decoder4 = DecoderBlock(256, encoder_channels[2], 128)
        self.decoder3 = DecoderBlock(128, encoder_channels[1], 64)
        self.decoder2 = DecoderBlock(64, encoder_channels[0], 32)
        self.decoder1 = DecoderBlock(32, 0, 16)
        
        # Final output
        self.final_conv = nn.Conv2d(16, num_classes, 1)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of encoder features [f1, f2, f3, f4, f5]
        
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        f1, f2, f3, f4, f5 = features
        
        d5 = self.decoder5(f5, f4)
        d4 = self.decoder4(d5, f3)
        d3 = self.decoder3(d4, f2)
        d2 = self.decoder2(d3, f1)
        d1 = self.decoder1(d2)
        
        out = self.final_conv(d1)
        
        return out


class OpticDiscSegmentationNetwork(nn.Module):
    """
    Network for Optic Disc Segmentation.
    
    Reference: Figure 3(a) - VGG16 Encoder + Simple Decoder
    
    "The optic disc is the entry point of vessels, and it is therefore crowded 
    with tangled vessels. Since estimating the vessel topology for such a region 
    is nearly impossible, we perform optic disc segmentation to exclude that 
    region from the subsequent procedures."
    
    Args:
        pretrained: Whether to use pretrained VGG16 weights
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        self.encoder = VGG16Encoder(pretrained=pretrained)
        self.decoder = SimpleDecoder(
            encoder_channels=self.encoder.out_channels,
            num_classes=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            OD segmentation logits [B, 1, H, W]
        """
        features = self.encoder(x)
        output = self.decoder(features)
        
        # Upsample to input size
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return output


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion module.
    
    Reference: Section 2.1.4 - "the final output is inferred from concatenated 
    multi-scale features of the VGG-16. Before the concatenation, feature maps 
    are resized to have identical spatial resolutions."
    """
    
    def __init__(
        self,
        encoder_channels: List[int] = [64, 128, 256, 512, 512],
        out_channels: int = 32
    ):
        super().__init__()
        
        # 1x1 convolutions to reduce channel dimensions
        self.reduce_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for c in encoder_channels
        ])
        
        # Fused output channels
        self.fused_channels = out_channels * len(encoder_channels)
    
    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Fuse multi-scale features.
        
        Args:
            features: List of encoder features
            target_size: Target spatial size (H, W)
            
        Returns:
            Fused features [B, fused_channels, H, W]
        """
        fused = []
        for feat, reduce_conv in zip(features, self.reduce_convs):
            reduced = reduce_conv(feat)
            resized = F.interpolate(reduced, size=target_size, mode='bilinear', align_corners=True)
            fused.append(resized)
        
        return torch.cat(fused, dim=1)


class MultiTaskDecoder(nn.Module):
    """
    Decoder for multi-task learning (vessel segmentation + AV classification).
    
    Reference: Figure 3(b) - Separate decoders for each task
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 3
    ):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, num_classes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class MultiTaskNetwork(nn.Module):
    """
    Multi-task Network for Binary Vessel Segmentation and AV Classification.
    
    Enhanced with Attention U-Net++ backbone for improved VCP performance.
    Reference: Screenshot - "Step 3 Enhanced VCP (Core) Backbone: Attention U-Net++"
    
    The loss function is defined as (Equation 2):
    L_segm_av = λ_segm * L_vsegm + λ_AV * L_AV
    
    Args:
        pretrained: Whether to use pretrained weights (only for VGG16 backbone)
        num_av_classes: Number of AV classes (default: 3 for background, artery, vein)
        multiscale_channels: Number of channels for multi-scale fusion
        use_attention_unet: Whether to use Attention U-Net++ backbone (default: True)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        num_av_classes: int = 3,
        multiscale_channels: int = 32,
        use_attention_unet: bool = False  # False = VGG16, True = Attention U-Net++
    ):
        super().__init__()

        self.use_attention_unet = use_attention_unet

        # Backbone selection
        if use_attention_unet:
            self.encoder = AttentionUNetPlusPlusEncoder(
                in_channels=3,
                base_channels=64,
                deep_supervision=False
            )
        else:
            self.encoder = VGG16Encoder(pretrained=pretrained)

        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(
            encoder_channels=self.encoder.out_channels,
            out_channels=multiscale_channels
        )
        
        fused_channels = self.feature_fusion.fused_channels
        
        # Task-specific decoders
        self.vessel_decoder = MultiTaskDecoder(
            in_channels=fused_channels,
            hidden_channels=128,
            num_classes=1  # Binary vessel segmentation
        )
        
        self.av_decoder = MultiTaskDecoder(
            in_channels=fused_channels,
            hidden_channels=128,
            num_classes=num_av_classes
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input image [B, 3, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            vessel_logits: [B, 1, H, W]
            av_logits: [B, num_classes, H, W]
            features (optional): [B, C, H, W]
        """
        B, _, H, W = x.shape
        
        # Encode
        encoder_features = self.encoder(x)
        
        # Fuse multi-scale features
        target_size = (H // 2, W // 2)  # Half resolution for efficiency
        fused_features = self.feature_fusion(encoder_features, target_size)
        
        # Decode
        vessel_logits = self.vessel_decoder(fused_features)
        av_logits = self.av_decoder(fused_features)
        
        # Upsample to input size
        vessel_logits = F.interpolate(vessel_logits, size=(H, W), mode='bilinear', align_corners=True)
        av_logits = F.interpolate(av_logits, size=(H, W), mode='bilinear', align_corners=True)
        
        if return_features:
            fused_features_upsampled = F.interpolate(fused_features, size=(H, W), mode='bilinear', align_corners=True)
            return vessel_logits, av_logits, fused_features_upsampled
        
        return vessel_logits, av_logits, None


class ThicknessOrientationEncoder(nn.Module):
    """
    Encoder for thickness and orientation classification.
    
    Enhanced with Attention U-Net++ backbone for improved VCP performance.
    Reference: Screenshot - "Step 3 Enhanced VCP (Core) Backbone: Attention U-Net++"
    
    Args:
        pretrained: Whether to use pretrained weights (only for VGG16 backbone)
        multiscale_channels: Number of channels for multi-scale fusion
        num_thickness_classes: Number of thickness classes
        num_orientation_classes: Number of orientation classes
        use_attention_unet: Whether to use Attention U-Net++ backbone (default: True)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        multiscale_channels: int = 32,
        num_thickness_classes: int = 5,
        num_orientation_classes: int = 7,
        use_attention_unet: bool = False  # False = VGG16, True = Attention U-Net++
    ):
        super().__init__()

        self.use_attention_unet = use_attention_unet

        # Backbone selection
        if use_attention_unet:
            self.encoder = AttentionUNetPlusPlusEncoder(
                in_channels=3,
                base_channels=64,
                deep_supervision=False
            )
        else:
            # Fallback to original VGG16 encoder
            self.encoder = VGG16Encoder(pretrained=pretrained)
        
        self.feature_fusion = MultiScaleFeatureFusion(
            encoder_channels=self.encoder.out_channels,
            out_channels=multiscale_channels
        )
        
        fused_channels = self.feature_fusion.fused_channels
        
        # Thickness classification head
        self.thickness_decoder = MultiTaskDecoder(
            in_channels=fused_channels,
            hidden_channels=128,
            num_classes=num_thickness_classes
        )
        
        # Orientation classification head
        self.orientation_decoder = MultiTaskDecoder(
            in_channels=fused_channels,
            hidden_channels=128,
            num_classes=num_orientation_classes
        )
        
        # Penultimate layer output (before final classification)
        # Used for connectivity prediction
        self.penultimate_channels = 64
        self.thickness_penultimate = nn.Sequential(
            nn.Conv2d(fused_channels, self.penultimate_channels, 3, padding=1),
            nn.BatchNorm2d(self.penultimate_channels),
            nn.ReLU(inplace=True)
        )
        self.orientation_penultimate = nn.Sequential(
            nn.Conv2d(fused_channels, self.penultimate_channels, 3, padding=1),
            nn.BatchNorm2d(self.penultimate_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_penultimate: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: Input image [B, 3, H, W]
            return_penultimate: Whether to return penultimate features
            
        Returns:
            thickness_logits: [B, num_thickness_classes, H, W]
            orientation_logits: [B, num_orientation_classes, H, W]
            thickness_features: [B, C, H, W] (if return_penultimate)
            orientation_features: [B, C, H, W] (if return_penultimate)
        """
        B, _, H, W = x.shape
        
        encoder_features = self.encoder(x)
        target_size = (H // 2, W // 2)
        fused_features = self.feature_fusion(encoder_features, target_size)
        
        # Classification outputs
        thickness_logits = self.thickness_decoder(fused_features)
        orientation_logits = self.orientation_decoder(fused_features)
        
        # Upsample to input size
        thickness_logits = F.interpolate(thickness_logits, size=(H, W), mode='bilinear', align_corners=True)
        orientation_logits = F.interpolate(orientation_logits, size=(H, W), mode='bilinear', align_corners=True)
        
        thickness_features = None
        orientation_features = None
        
        if return_penultimate:
            thickness_features = self.thickness_penultimate(fused_features)
            orientation_features = self.orientation_penultimate(fused_features)
            thickness_features = F.interpolate(thickness_features, size=(H, W), mode='bilinear', align_corners=True)
            orientation_features = F.interpolate(orientation_features, size=(H, W), mode='bilinear', align_corners=True)
        
        return thickness_logits, orientation_logits, thickness_features, orientation_features


class HadamardProductLayer(nn.Module):
    """
    Hadamard Product Layer for symmetric pairwise feature fusion.
    
    Reference: Section 2.1.4 and Figure 3(c) - "A Hadamard product symbolized by ⊙ 
    is used on the 256-dim outputs of a fully connected layer applied on the 
    concatenated thickness/orientation features."
    
    "To ensure the symmetry of this pairwise classification, a Hadamard product 
    symbolized by ⊙ is used on the 256-dim outputs of a fully connected layer... 
    One of the desired properties for the pairwise classification is to satisfy 
    symmetry, which means that the output is the same regardless of the input order."
    
    The Hadamard (element-wise) product ensures:
    f(a, b) = f(b, a)
    """
    
    def __init__(self, in_features: int = 256):
        super().__init__()
        self.in_features = in_features
    
    def forward(self, features_i: torch.Tensor, features_j: torch.Tensor) -> torch.Tensor:
        """
        Compute Hadamard product of two feature vectors.
        
        Args:
            features_i: Features for pixel/segment i [B, D]
            features_j: Features for pixel/segment j [B, D]
            
        Returns:
            Symmetric combined features [B, D]
        """
        # Element-wise product ensures symmetry
        return features_i * features_j


class ConnectivityNetwork(nn.Module):
    """
    Connectivity Classification Network.
    
    Reference: Figure 3(c) - "Network for vascular connectivity classification, 
    including multi-task auxiliary outputs for thickness/orientation classifications."
    
    Architecture:
    1. Input: Features from thickness and orientation encoders
    2. For each pixel pair (i, j):
       - Extract features at positions i and j
       - Concatenate thickness and orientation features
       - Process through FC layers
       - Apply Hadamard product for symmetry
       - Final FC layer for binary connectivity prediction
    
    The connectivity GT label for training is decided from the rightmost topology GT.
    
    Args:
        feature_channels: Number of input feature channels (thickness + orientation)
        fc_units: Number of units in FC layers (default: 256 per paper)
    """
    
    def __init__(
        self,
        feature_channels: int = 128,  # 64 thickness + 64 orientation
        fc_units: int = 256
    ):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.fc_units = fc_units
        
        # FC layers before Hadamard product
        # Takes concatenated features from pixel i (thickness + orientation)
        self.fc_before_hadamard = nn.Sequential(
            nn.Linear(feature_channels, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Hadamard product layer
        self.hadamard = HadamardProductLayer(fc_units)
        
        # FC layers after Hadamard product
        self.fc_after_hadamard = nn.Sequential(
            nn.Linear(fc_units, fc_units // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_units // 2, 2)  # Binary: connected vs unconnected
        )
    
    def forward(
        self,
        thickness_features: torch.Tensor,
        orientation_features: torch.Tensor,
        coords_i: torch.Tensor,
        coords_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict connectivity between pixel pairs.
        
        Args:
            thickness_features: [B, C_t, H, W] thickness features
            orientation_features: [B, C_o, H, W] orientation features
            coords_i: [B, 2] coordinates (y, x) for pixel i
            coords_j: [B, 2] coordinates (y, x) for pixel j
            
        Returns:
            connectivity_logits: [B, 2] logits for (unconnected, connected)
        """
        B = thickness_features.shape[0]

        # Concatenate thickness and orientation features
        combined_features = torch.cat([thickness_features, orientation_features], dim=1)  # [B, C, H, W]
        H, W = combined_features.shape[2:]

        # Clamp and extract features for all batch elements at once (no Python loop)
        yi = coords_i[:, 0].long().clamp(0, H - 1)  # [B]
        xi = coords_i[:, 1].long().clamp(0, W - 1)  # [B]
        yj = coords_j[:, 0].long().clamp(0, H - 1)  # [B]
        xj = coords_j[:, 1].long().clamp(0, W - 1)  # [B]

        b_idx = torch.arange(B, device=combined_features.device)
        features_i = combined_features[b_idx, :, yi, xi]  # [B, C]
        features_j = combined_features[b_idx, :, yj, xj]  # [B, C]
        
        # FC layers before Hadamard
        emb_i = self.fc_before_hadamard(features_i)  # [B, fc_units]
        emb_j = self.fc_before_hadamard(features_j)  # [B, fc_units]
        
        # Hadamard product for symmetry
        combined = self.hadamard(emb_i, emb_j)  # [B, fc_units]
        
        # Final classification
        logits = self.fc_after_hadamard(combined)  # [B, 2]
        
        return logits
    
    def forward_batch(
        self,
        combined_features: torch.Tensor,
        coords_i: torch.Tensor,
        coords_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch version for efficiency when features are pre-computed.
        
        Args:
            combined_features: [B, C, H, W] concatenated thickness + orientation features
            coords_i: [B, N, 2] coordinates for N pairs, pixel i
            coords_j: [B, N, 2] coordinates for N pairs, pixel j
            
        Returns:
            connectivity_logits: [B, N, 2]
        """
        B, N, _ = coords_i.shape
        C, H, W = combined_features.shape[1:]

        # Clamp all coordinates at once
        yi = coords_i[..., 0].long().clamp(0, H - 1)  # [B, N]
        xi = coords_i[..., 1].long().clamp(0, W - 1)  # [B, N]
        yj = coords_j[..., 0].long().clamp(0, H - 1)  # [B, N]
        xj = coords_j[..., 1].long().clamp(0, W - 1)  # [B, N]

        # Build batch indices [B, N] for advanced indexing
        b_idx = torch.arange(B, device=combined_features.device).unsqueeze(1).expand(B, N)  # [B, N]

        # Extract features for all pairs in all batches at once
        feat_i = combined_features[b_idx, :, yi, xi]  # [B, N, C]
        feat_j = combined_features[b_idx, :, yj, xj]  # [B, N, C]

        # Flatten batch and pair dims, run FC + Hadamard, then reshape
        feat_i = feat_i.view(B * N, C)
        feat_j = feat_j.view(B * N, C)

        emb_i = self.fc_before_hadamard(feat_i)           # [B*N, fc_units]
        emb_j = self.fc_before_hadamard(feat_j)           # [B*N, fc_units]
        combined = self.hadamard(emb_i, emb_j)            # [B*N, fc_units]
        logits = self.fc_after_hadamard(combined)          # [B*N, 2]

        return logits.view(B, N, 2)  # [B, N, 2]


class FullConnectivityPipeline(nn.Module):
    """
    Full pipeline combining thickness/orientation encoder with connectivity network.
    
    Enhanced with Attention U-Net++ backbone for improved VCP performance.
    Reference: Screenshot - "Step 3 Enhanced VCP (Core) Backbone: Attention U-Net++"
    
    Reference: Equation 3 - "L_thick_ori_conn = λ_thick * L_thick + λ_ori * L_ori + λ_conn * L_conn"
    
    Args:
        pretrained: Whether to use pretrained weights (only for VGG16 backbone)
        multiscale_channels: Number of channels for multi-scale fusion
        num_thickness_classes: Number of thickness classes
        num_orientation_classes: Number of orientation classes
        fc_units: Number of FC units
        use_attention_unet: Whether to use Attention U-Net++ backbone (default: True)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        multiscale_channels: int = 32,
        num_thickness_classes: int = 5,
        num_orientation_classes: int = 7,
        fc_units: int = 256,
        use_attention_unet: bool = False  # False = VGG16, True = Attention U-Net++
    ):
        super().__init__()
        
        # Thickness/Orientation encoder with Attention U-Net++ backbone
        self.thick_ori_encoder = ThicknessOrientationEncoder(
            pretrained=pretrained,
            multiscale_channels=multiscale_channels,
            num_thickness_classes=num_thickness_classes,
            num_orientation_classes=num_orientation_classes,
            use_attention_unet=use_attention_unet
        )
        
        # Connectivity network
        feature_channels = 2 * self.thick_ori_encoder.penultimate_channels
        self.connectivity_net = ConnectivityNetwork(
            feature_channels=feature_channels,
            fc_units=fc_units
        )
    
    def forward(
        self,
        x: torch.Tensor,
        coords_i: Optional[torch.Tensor] = None,
        coords_j: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input image [B, 3, H, W]
            coords_i: [B, 2] or [B, N, 2] pixel coordinates for pair i
            coords_j: [B, 2] or [B, N, 2] pixel coordinates for pair j
            
        Returns:
            thickness_logits: [B, num_thickness_classes, H, W]
            orientation_logits: [B, num_orientation_classes, H, W]
            connectivity_logits: [B, 2] or [B, N, 2] (if coords provided)
        """
        # Get thickness/orientation predictions and features
        thick_logits, ori_logits, thick_feat, ori_feat = self.thick_ori_encoder(
            x, return_penultimate=True
        )
        
        connectivity_logits = None
        
        if coords_i is not None and coords_j is not None:
            if coords_i.dim() == 2:
                # Single pair per image
                connectivity_logits = self.connectivity_net(
                    thick_feat, ori_feat, coords_i, coords_j
                )
            else:
                # Multiple pairs per image
                combined = torch.cat([thick_feat, ori_feat], dim=1)
                connectivity_logits = self.connectivity_net.forward_batch(
                    combined, coords_i, coords_j
                )
        
        return thick_logits, ori_logits, connectivity_logits
    
    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get penultimate features for external connectivity prediction.
        
        Returns:
            thickness_features: [B, C, H, W]
            orientation_features: [B, C, H, W]
        """
        _, _, thick_feat, ori_feat = self.thick_ori_encoder(x, return_penultimate=True)
        return thick_feat, ori_feat
