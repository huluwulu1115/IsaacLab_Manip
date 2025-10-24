# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Vision-based student policy network with RGBD encoder for distillation.

This module provides CNN encoders for processing RGBD camera images
and integrating them with proprioceptive observations.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SimpleCNNEncoder(nn.Module):
    """
    Simple CNN encoder for RGBD images (flexible resolution).
    
    Architecture:
    - Input: (batch, 4, H, W) - RGBD
    - 4 Conv layers with stride=2 (total /16 downsampling)
    - Output: (batch, feature_dim) - flattened features
    
    Resolution: Automatically adapts to input size
    """
    
    def __init__(self, feature_dim: int = 256, input_height: int = 240, input_width: int = 320):
        super().__init__()
        
        # Input: 4 channels (RGB + Depth)
        # Each conv layer with stride=2 reduces size by half
        # Total reduction: /16 (4 layers of stride=2)
        
        self.encoder = nn.Sequential(
            # Conv1: /2
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Conv2: /4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv3: /8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Conv4: /16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Calculate output size after 4 stride=2 convolutions
        # Formula: output_size = ceil(input_size / 16)
        import math
        out_height = math.ceil(input_height / 16)
        out_width = math.ceil(input_width / 16)
        self.flatten_size = 256 * out_height * out_width
        
        print(f"[CNN Encoder] Input: {input_height}×{input_width} → Output: {out_height}×{out_width} → Flatten: {self.flatten_size}")
        
        # Fully connected layer to compress to feature_dim
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, feature_dim),
        )
        
    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rgbd: (batch, height, width, 4) - RGBD image
            
        Returns:
            features: (batch, feature_dim) - encoded features
        """
        # Permute from (B, H, W, C) to (B, C, H, W)
        x = rgbd.permute(0, 3, 1, 2).contiguous()
        
        # Normalize RGB to [0, 1] and depth to reasonable range
        rgb = x[:, :3, :, :] / 255.0
        depth = x[:, 3:4, :, :].clamp(0.01, 10.0) / 10.0  # Normalize to [0, 1]
        x = torch.cat([rgb, depth], dim=1)
        
        # CNN encoder
        x = self.encoder(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC compression
        features = self.fc(x)
        
        return features


class VisionStudentPolicy(nn.Module):
    """
    Vision-based student policy combining RGBD encoder with MLP.
    
    Input:
    - Proprioception: (batch, 24) - joint states + actions
    - RGBD image: (batch, H, W, 4) - flexible resolution
    
    Output:
    - Action: (batch, action_dim)
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
    ):
        super().__init__()
        
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.vision_feature_dim = vision_feature_dim
        
        # Vision encoder with configurable resolution
        self.vision_encoder = SimpleCNNEncoder(
            feature_dim=vision_feature_dim,
            input_height=image_height,
            input_width=image_width
        )
        
        # MLP for combined features
        # Input: proprio (24) + vision features (256) = 280
        mlp_input_dim = proprio_dim + vision_feature_dim
        
        # Activation function
        if activation == "elu":
            act_fn = nn.ELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP
        layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn(inplace=True))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Action distribution parameters
        self.action_mean = nn.Parameter(torch.zeros(action_dim), requires_grad=False)
        self.action_std = nn.Parameter(torch.ones(action_dim), requires_grad=True)
        
    def forward(self, proprio: torch.Tensor, rgbd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            proprio: (batch, proprio_dim) - proprioceptive observations
            rgbd: (batch, height, width, 4) - RGBD image
            
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
        
        # Expand action_std to batch size
        action_std = self.action_std.unsqueeze(0).expand_as(action_mean)
        
        return action_mean, action_std
    
    def act(self, proprio: torch.Tensor, rgbd: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample action from policy.
        
        Args:
            proprio: (batch, proprio_dim)
            rgbd: (batch, height, width, 4)
            deterministic: If True, return mean action
            
        Returns:
            action: (batch, action_dim)
        """
        action_mean, action_std = self.forward(proprio, rgbd)
        
        if deterministic:
            return action_mean
        else:
            # Sample from Gaussian
            noise = torch.randn_like(action_mean)
            return action_mean + action_std * noise
    
    def evaluate(
        self, 
        proprio: torch.Tensor, 
        rgbd: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.
        
        Args:
            proprio: (batch, proprio_dim)
            rgbd: (batch, height, width, 4)
            actions: (batch, action_dim) - actions to evaluate
            
        Returns:
            log_prob: (batch,) - log probability of actions
            entropy: (batch,) - policy entropy
        """
        action_mean, action_std = self.forward(proprio, rgbd)
        
        # Gaussian log probability
        var = action_std ** 2
        log_prob = -0.5 * (
            ((actions - action_mean) ** 2) / var +
            torch.log(2 * torch.pi * var)
        ).sum(dim=-1)
        
        # Entropy of Gaussian
        entropy = 0.5 * (torch.log(2 * torch.pi * var) + 1).sum(dim=-1)
        
        return log_prob, entropy


class VisionStudentTeacherWrapper(nn.Module):
    """
    Wrapper combining vision student policy with frozen teacher policy.
    
    This is used for distillation training where the teacher provides
    action labels based on privileged state observations.
    """
    
    def __init__(
        self,
        student: VisionStudentPolicy,
        teacher: nn.Module,
        freeze_teacher: bool = True,
    ):
        super().__init__()
        
        self.student = student
        self.teacher = teacher
        
        # Freeze teacher parameters
        if freeze_teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()
    
    def student_forward(self, proprio: torch.Tensor, rgbd: torch.Tensor) -> torch.Tensor:
        """Get student action (for deployment)."""
        return self.student.act(proprio, rgbd, deterministic=False)
    
    def teacher_forward(self, teacher_obs: torch.Tensor) -> torch.Tensor:
        """Get teacher action (for labeling)."""
        with torch.no_grad():
            # Assume teacher is a standard ActorCritic from RSL-RL
            return self.teacher.act(teacher_obs, deterministic=True)
    
    def distillation_loss(
        self,
        proprio: torch.Tensor,
        rgbd: torch.Tensor,
        teacher_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss (MSE between student and teacher actions).
        
        Args:
            proprio: Student observations (24D)
            rgbd: RGBD images
            teacher_obs: Teacher observations (41D)
            
        Returns:
            loss: MSE loss
        """
        # Get student action
        student_action = self.student.act(proprio, rgbd, deterministic=False)
        
        # Get teacher action (labels)
        teacher_action = self.teacher_forward(teacher_obs)
        
        # MSE loss
        loss = nn.functional.mse_loss(student_action, teacher_action)
        
        return loss

