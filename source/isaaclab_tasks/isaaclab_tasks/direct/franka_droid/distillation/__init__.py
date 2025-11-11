# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Distillation package for vision-based policy learning.

This package contains all components needed for teacher-student distillation:
- Environment configuration with vision observations
- Vision encoders (SimpleCNN, ResNet18, Transformer)
- Student policy networks
- Distillation agent (Dextrah-style online DAgger)
- Episode reward tracking utilities
"""

from .distillation_env import FrankaDroidDistillationEnv, FrankaDroidDistillationEnvCfg
from .vision_policy_resnet import VisionStudentPolicyResNet
from .resnet_encoder import ResNet18RGBDEncoder, DexterahStyleEncoder
from .rsl_rl_distillation_cfg import FrankaDroidDistillationRunnerCfg

__all__ = [
    # Environment
    "FrankaDroidDistillationEnv",
    "FrankaDroidDistillationEnvCfg",
    # RSL-RL Config
    "FrankaDroidDistillationRunnerCfg",
    # Student Policies
    "VisionStudentPolicyResNet",
    # Vision Encoders
    "ResNet18RGBDEncoder",
    "DexterahStyleEncoder",
]
