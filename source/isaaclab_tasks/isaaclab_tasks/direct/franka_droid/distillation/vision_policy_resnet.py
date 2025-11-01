# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Vision policy with ResNet18 encoder (DEXTRAH-aligned).

Uses stronger ResNet18 encoder to replace SimpleCNN.
"""

import torch
import torch.nn as nn
from typing import Tuple

from .resnet_encoder import ResNet18RGBDEncoder, DexterahStyleEncoder


class VisionStudentPolicyResNet(nn.Module):
    """
    Vision-based student policy using ResNet18 encoder.
    
    Input:
    - Proprioception: (batch, 24)
    - RGBD: (batch, H, W, 4)
    
    Output:
    - Actions: (batch, 8)
    """
    
    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        vision_feature_dim: int = 256,
        hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        image_height: int = 240,
        image_width: int = 320,
        encoder_type: str = "dextrah",  # "resnet18" or "dextrah"
        use_pretrained: bool = True,
    ):
        super().__init__()
        
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.vision_feature_dim = vision_feature_dim
        self.encoder_type = encoder_type
        
        # Vision encoder (select encoder type)
        if encoder_type == "dextrah":
            # Full DEXTRAH-style: ResNet18 + Transformer
            self.vision_encoder = DexterahStyleEncoder(
                feature_dim=vision_feature_dim,
                input_height=image_height,
                input_width=image_width,
                n_transformer_heads=4,
                n_transformer_layers=2,
            )
            print(f"[Student Policy] Using DEXTRAH-style encoder (ResNet18 + Transformer)")
        else:
            # Simplified ResNet18 (without Transformer)
            self.vision_encoder = ResNet18RGBDEncoder(
                feature_dim=vision_feature_dim,
                input_height=image_height,
                input_width=image_width,
                use_pretrained=use_pretrained,
                freeze_backbone=False,
            )
            print(f"[Student Policy] Using ResNet18 encoder (pretrained={use_pretrained})")
        
        # MLP for combined features
        mlp_input_dim = proprio_dim + vision_feature_dim
        
        if activation == "elu":
            act_fn = nn.ELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP
        layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Action distribution parameters (DEXTRAH-style)
        # Use log_std instead of std directly to prevent collapse
        # log_std = 0 → std = exp(0) = 1.0 (good initial exploration)
        self.action_mean = nn.Parameter(torch.zeros(action_dim), requires_grad=False)
        self.log_action_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        
        print(f"[Student Policy] MLP: {mlp_input_dim} → {hidden_dims} → {action_dim}")
        print(f"[Student Policy] Using log_std (DEXTRAH-style, prevents collapse)")
        print(f"[Student Policy] Initial std: {torch.exp(self.log_action_std).mean().item():.3f}")
        print(f"[Student Policy] Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, proprio: torch.Tensor, rgbd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            proprio: (batch, proprio_dim)
            rgbd: (batch, height, width, 4)
            
        Returns:
            action_mean: (batch, action_dim)
            action_std: (batch, action_dim)
        """
        # Encode vision
        vision_features = self.vision_encoder(rgbd)
        
        # Concatenate proprio + vision
        combined = torch.cat([proprio, vision_features], dim=-1)
        
        # MLP
        action_mean = self.mlp(combined)
        
        # DEXTRAH-style: exp(log_std) to ensure std > 0 and prevent collapse
        # No clamp - let sigma_loss guide the value to match teacher (σ ≈ 0.8-4.0)
        action_std = torch.exp(self.log_action_std)
        action_std = action_std.unsqueeze(0).expand_as(action_mean)
        
        return action_mean, action_std
    
    def act(self, proprio: torch.Tensor, rgbd: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action."""
        action_mean, action_std = self.forward(proprio, rgbd)
        
        if deterministic:
            return action_mean
        else:
            noise = torch.randn_like(action_mean)
            return action_mean + action_std * noise

