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
        
        # Remove avgpool and fc (use Sequential to properly remove layers)
        # This keeps only conv layers + bn + relu, outputs (B, 512, H/32, W/32)
        self.resnet = nn.Sequential(*list(resnet18.children())[:-2])
        
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

