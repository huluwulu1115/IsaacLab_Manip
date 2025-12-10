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

    # Teacher policy (state-based) - MUST match trained teacher exactly!
    teacher_obs_normalization: bool = True  # Match teacher training
    teacher_hidden_dims: list[int] = [512, 256, 128]  # Match rsl_rl_ppo_cfg.py

    activation: str = "elu"

    # Vision encoder configuration
    use_vision_encoder: bool = True
    vision_encoder_type: str = "resnet18"  # or "dextrah"
    vision_feature_dim: int = 256  # Compressed vision features


@configclass
class FrankaDroidDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """
    Online DAgger distillation configuration (DEXTRAH-aligned).

    Teacher policy: State-based (41D privileged state)
    Student policy: Vision-based (24D proprioception + RGBD camera)

    Key features:
    - Per-step updates (true online learning)
    - Beta scheduling (teacher→student transition)
    - Extended schedule based on empirical results
    """

    class_name: str = "DistillationRunner"

    # Training parameters
    num_steps_per_env: int = 24  # Steps per iteration (for logging)
    max_iterations: int = 1000  # Distillation iterations
    save_interval: int = 50

    # Beta schedule (extended based on empirical observation)
    # Success rate drops too fast with original schedule
    beta_start_decay: int = 15_000  # Start linear decay (延后)
    beta_end_decay: int = 60_000  # End decay (延长，给更多学习时间)
    beta_strategy: str = "linear"  # "step" or "linear"

    # Experiment name
    experiment_name: str = "franka_droid_distillation"
    run_name: str = "vision_student"

    # Observation groups for distillation
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

    # Teacher checkpoint configuration
    resume: bool = False  # Set to False for distillation
    load_run: str = "2025-10-23_19-43-40"  # Teacher run directory
    load_checkpoint: str = "model_1499.pt"  # Teacher checkpoint

    # Teacher policy configuration - MUST match rsl_rl_ppo_cfg.py exactly!
    teacher_config: dict = {
        "num_observations": 41,  # Privileged state observations
        "num_actions": 8,  # Robot actions
        "actor_hidden_dims": [512, 256, 128],  # Must match rsl_rl_ppo_cfg.py
        "activation": "elu",  # Must match teacher training
    }

    # Clip actions (match teacher)
    clip_actions: bool = True

    # Logging
    logger: str = "wandb"  # or "wandb"
    wandb_project: str = "franka_droid_distillation"

    # Empirical normalization
    empirical_normalization: bool = True
