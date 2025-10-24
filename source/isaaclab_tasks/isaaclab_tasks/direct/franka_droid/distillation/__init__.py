# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Distillation package for vision-based policy learning.

This package contains all components needed for teacher-student distillation:
- Environment configuration with vision observations
- Vision encoders (SimpleCNN, ResNet18)
- Student policy networks
- RSL-RL distillation configuration
- Episode reward tracking utilities
"""

from .distillation_env import FrankaDroidDistillationEnv, FrankaDroidDistillationEnvCfg
from .rsl_rl_distillation_cfg import FrankaDroidDistillationRunnerCfg, FrankaDroidVisionStudentTeacherCfg
from .vision_policy_network import VisionStudentPolicy, SimpleCNNEncoder
from .vision_policy_resnet import VisionStudentPolicyResNet
from .resnet_encoder import ResNet18RGBDEncoder, DexterahStyleEncoder
from .episode_reward_tracker import EpisodeRewardTracker

__all__ = [
    # Environment
    "FrankaDroidDistillationEnv",
    "FrankaDroidDistillationEnvCfg",
    # Configuration
    "FrankaDroidDistillationRunnerCfg",
    "FrankaDroidVisionStudentTeacherCfg",
    # Policies
    "VisionStudentPolicy",
    "VisionStudentPolicyResNet",
    # Encoders
    "SimpleCNNEncoder",
    "ResNet18RGBDEncoder",
    "DexterahStyleEncoder",
    # Utils
    "EpisodeRewardTracker",
]

