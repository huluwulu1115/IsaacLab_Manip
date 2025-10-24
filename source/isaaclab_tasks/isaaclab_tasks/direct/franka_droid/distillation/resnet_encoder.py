# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ResNet-based RGBD encoder (DEXTRAH-style).

Reference implementation based on DEXTRAH's mono_encoder.py.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18RGBDEncoder(nn.Module):
    """
    ResNet18-based encoder for RGBD images.
    
    Architecture (based on DEXTRAH):
    - ResNet18 backbone (pretrained ImageNet weights)
    - Modified first layer to accept 4-channel input (RGBD)
    - Removed final avgpool and fc layers
    - Added spatial flatten and projection
    
    Input: (batch, 240, 320, 4) - RGBD
    Output: (batch, feature_dim) - Compressed features
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        input_height: int = 240,
        input_width: int = 320,
        use_pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.use_pretrained = use_pretrained
        
        # Load ResNet18
        if use_pretrained:
            resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("[ResNet18 Encoder] Using ImageNet pretrained weights")
        else:
            resnet18 = models.resnet18(weights=None)
            print("[ResNet18 Encoder] Training from scratch")
        
        # Modify first layer: 3 channels → 4 channels (RGBD)
        original_conv1 = resnet18.conv1
        resnet18.conv1 = nn.Conv2d(
            4,  # RGBD instead of RGB
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # If using pretrained weights, copy RGB weights and initialize depth channel
        if use_pretrained:
            with torch.no_grad():
                # Copy RGB weights
                resnet18.conv1.weight[:, :3, :, :] = original_conv1.weight
                # Initialize depth channel as mean of RGB channels
                resnet18.conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
        
        # Remove final avgpool and fc (we need spatial features)
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2])
        # Output: (batch, 512, H', W') - H' and W' depend on input size
        
        # Freeze backbone (optional)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[ResNet18 Encoder] Backbone frozen")
        
        # DEXTRAH-style spatial pooling: pool to fixed size instead of (1,1)
        # This preserves spatial information, similar to DEXTRAH's (8, 16) approach
        # For 240×320 input, use (8, 16) to preserve 128 spatial tokens
        self.spatial_pool_size = (8, 16)  # 8*16 = 128 tokens
        self.downproject = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),  # Reduce channels 512→128
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(self.spatial_pool_size),  # Fixed spatial size
        )
        
        # Projection layers (DEXTRAH-style)
        # flatten_size = 128 channels × 8 × 16 = 16384
        flatten_size = 128 * self.spatial_pool_size[0] * self.spatial_pool_size[1]
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, feature_dim),
        )
        
        print(f"[ResNet18 Encoder] Input: {input_height}×{input_width}")
        print(f"[ResNet18 Encoder] DEXTRAH-style adaptive pooling to {self.spatial_pool_size}")
        print(f"[ResNet18 Encoder] Spatial tokens: {flatten_size} (128×{self.spatial_pool_size[0]}×{self.spatial_pool_size[1]})")
        print(f"[ResNet18 Encoder] Final features: {feature_dim}D")
    
    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rgbd: (batch, height, width, 4) - RGBD image
            
        Returns:
            features: (batch, feature_dim)
        """
        # Permute from (B, H, W, C) to (B, C, H, W)
        x = rgbd.permute(0, 3, 1, 2).contiguous()
        
        # Normalize RGB to [0, 1] (DEXTRAH uses ImageNet normalization)
        rgb = x[:, :3, :, :] / 255.0
        
        # ImageNet normalization for RGB
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        rgb = (rgb - mean) / std
        
        # Normalize depth to [0, 1]
        depth = x[:, 3:4, :, :].clamp(0.01, 10.0) / 10.0
        
        # Concatenate normalized RGB + Depth
        x = torch.cat([rgb, depth], dim=1)
        
        # ResNet backbone
        features = self.backbone(x)  # (B, 512, H', W')
        
        # DEXTRAH-style downproject: channel reduction + spatial pooling
        features = self.downproject(features)  # (B, 128, 8, 16)
        
        # Projection to feature_dim
        features = self.projection(features)  # (B, feature_dim)
        
        return features


class DexterahStyleEncoder(nn.Module):
    """
    Full DEXTRAH-style encoder: ResNet18 + Transformer.
    
    This is the implementation based on DEXTRAH's mono_encoder.py.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        input_height: int = 240,
        input_width: int = 320,
        n_transformer_heads: int = 4,
        n_transformer_layers: int = 2,
    ):
        super().__init__()
        
        # ResNet18 backbone
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify first layer to 4 channels
        original_conv1 = resnet18.conv1
        resnet18.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet18.conv1.weight[:, :3, :, :] = original_conv1.weight
            resnet18.conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
        
        # Remove avgpool and fc
        resnet18.avgpool = nn.Identity()  # type: ignore
        resnet18.fc = nn.Identity()  # type: ignore
        self.resnet = resnet18
        
        # Spatial dimensions after ResNet18
        out_h = input_height // 32
        out_w = input_width // 32
        num_tokens = out_h * out_w
        
        # Transformer (DEXTRAH-style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,  # ResNet18 output channels
            nhead=n_transformer_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, feature_dim),
        )
        
        print(f"[DEXTRAH Encoder] ResNet18 + {n_transformer_layers}-layer Transformer")
        print(f"[DEXTRAH Encoder] Spatial tokens: {num_tokens} ({out_h}×{out_w})")
        print(f"[DEXTRAH Encoder] Output: {feature_dim}D")
    
    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        """Forward with transformer."""
        # Permute and normalize (same as ResNet18RGBDEncoder)
        x = rgbd.permute(0, 3, 1, 2).contiguous()
        
        rgb = x[:, :3, :, :] / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        rgb = (rgb - mean) / std
        
        depth = x[:, 3:4, :, :].clamp(0.01, 10.0) / 10.0
        x = torch.cat([rgb, depth], dim=1)
        
        # ResNet feature extraction
        x = self.resnet(x)  # (B, 512, H/32, W/32)
        
        # Reshape for transformer: (B, 512, H, W) → (B, H*W, 512)
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(0, 2, 1)  # (B, H*W, 512)
        
        # Transformer
        x = self.transformer(x)  # (B, H*W, 512)
        
        # Use CLS token (first token)
        x = x[:, 0, :]  # (B, 512)
        
        # Output projection
        features = self.output_proj(x)  # (B, feature_dim)
        
        return features

