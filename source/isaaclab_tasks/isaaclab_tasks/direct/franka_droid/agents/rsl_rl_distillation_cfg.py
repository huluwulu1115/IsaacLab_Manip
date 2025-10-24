# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RSL-RL Distillation configuration for Franka DROID environment.

Online DAgger: Teacher policy (state-based) -> Student policy (vision-based RGBD)
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class FrankaDroidVisionStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    """Vision-based student-teacher configuration with RGBD camera support."""
    
    class_name: str = "StudentTeacher"
    
    # Student policy (vision-based)
    init_noise_std: float = 0.1
    noise_std_type: str = "scalar"
    student_obs_normalization: bool = False  # Vision features shouldn't be normalized
    student_hidden_dims: list[int] = [512, 256, 128]  # Larger network for vision
    
    # Teacher policy (state-based) - must match trained teacher
    teacher_obs_normalization: bool = True  # Match teacher training
    teacher_hidden_dims: list[int] = [256, 128, 64]  # Match teacher training
    
    activation: str = "elu"
    
    # Vision encoder configuration (for RGBD processing)
    # RGB: (H=360, W=640, C=3) -> flatten to 691,200
    # Depth: (H=360, W=640, C=1) -> flatten to 230,400
    # Total: 921,600 dim (too large, need CNN encoder!)
    
    # Simple CNN encoder for RGBD
    use_vision_encoder: bool = True
    vision_encoder_type: str = "simple_cnn"  # or "resnet18"
    vision_feature_dim: int = 256  # Compressed vision features
    
    # Input: RGB (3 channels) + Depth (1 channel) = 4 channels
    # Output: vision_feature_dim


@configclass  
class FrankaDroidDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """
    Online DAgger distillation configuration.
    
    Teacher policy: State-based (41D privileged state)
    Student policy: Vision-based (24D proprioception + RGBD camera)
    """
    
    class_name: str = "DistillationRunner"
    
    # Training parameters
    num_steps_per_env: int = 24  # Match teacher training
    max_iterations: int = 1000  # Distillation iterations
    save_interval: int = 50
    
    # Experiment name
    experiment_name: str = "franka_droid_distillation"
    run_name: str = "vision_student"
    
    # Observation groups for distillation
    # "policy" = student observations (24D proprioception)
    # "teacher" = teacher observations (41D state) for teacher labeling
    obs_groups: dict = {
        "policy": ["policy"],  # Student uses proprio only
        "teacher": ["teacher_obs"],  # Teacher uses full state
    }
    
    # Policy configuration
    policy: FrankaDroidVisionStudentTeacherCfg = FrankaDroidVisionStudentTeacherCfg()
    
    # Distillation algorithm (Online DAgger)
    algorithm: RslRlDistillationAlgorithmCfg = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,  # Number of gradient updates per iteration
        learning_rate=1e-4,  # Lower LR for stable vision learning
        gradient_length=24,  # Match num_steps_per_env
        max_grad_norm=1.0,  # Gradient clipping
        optimizer="adam",
        loss_type="mse",  # Mean squared error between student/teacher actions
    )
    
    # Device
    device: str = "cuda:0"
    seed: int = 42
    
    # Resume from teacher checkpoint
    resume: bool = False  # Set to False for distillation (we load teacher separately)
    load_run: str = "2025-10-23_19-43-40"  # Teacher run directory
    load_checkpoint: str = "model_1499.pt"  # Teacher checkpoint
    
    # Clip actions (match teacher)
    clip_actions: bool = True
    
    # Logging
    logger: str = "tensorboard"  # or "wandb"
    wandb_project: str = "isaaclab_droid"
    
    # Empirical normalization (use teacher's normalization stats)
    empirical_normalization: bool = True

