#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Vision-based policy distillation training script using Online DAgger.

This script trains a vision-based student policy by distilling knowledge from
a state-based teacher policy using RGBD camera observations.

Usage:
    python train_vision_distillation.py --task Isaac-DROID-Distillation-v0 \
        --num_envs 2048 --load_run 2025-10-23_19-43-40 --checkpoint model_1499.pt
"""

import argparse
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path

# IsaacLab imports
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train vision-based student policy via distillation")
parser.add_argument("--task", type=str, default="Isaac-DROID-Distillation-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments")
parser.add_argument("--seed", type=str, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=1000, help="Number of distillation iterations")
parser.add_argument("--load_run", type=str, required=True, help="Teacher run directory name")
parser.add_argument("--checkpoint", type=str, default="model_1499.pt", help="Teacher checkpoint filename")
parser.add_argument("--video", action="store_true", help="Record video")

# Append AppLauncher arguments (this adds --headless, --device, etc. automatically)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# CRITICAL: Enable cameras for RGBD vision observations
args_cli.enable_cameras = True

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.franka_droid.agents.vision_policy_network import (
    VisionStudentPolicy,
    VisionStudentTeacherWrapper,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def load_teacher_policy(checkpoint_path: str, device: str) -> nn.Module:
    """
    Manually load teacher policy from checkpoint.
    
    Teacher policy is a simple MLP that takes 41D state and outputs 8D actions.
    We only need the actor network for inference (no critic needed for distillation).
    """
    print(f"\n[INFO] Loading teacher policy from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    # Create a simple teacher actor network manually
    class TeacherActor(nn.Module):
        def __init__(self):
            super().__init__()
            # Match the trained teacher architecture: [256, 128, 64]
            self.network = nn.Sequential(
                nn.Linear(41, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 8),
            )
            # Standard deviation for action noise
            self.std = nn.Parameter(torch.ones(8), requires_grad=False)
            
        def forward(self, obs):
            return self.network(obs)
        
        def act(self, obs, deterministic=True):
            """Teacher inference method."""
            action_mean = self.forward(obs)
            if deterministic:
                return action_mean
            else:
                return action_mean + torch.randn_like(action_mean) * self.std
    
    # Create teacher
    teacher = TeacherActor().to(device)
    
    # Load weights from checkpoint
    # Extract only actor weights
    actor_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("actor.") or k == "std":
            # Remove "actor." prefix for our simple network
            new_key = k.replace("actor.", "network.")
            actor_state_dict[new_key] = v
        if k == "std":
            actor_state_dict["std"] = v
    
    teacher.load_state_dict(actor_state_dict, strict=False)
    teacher.eval()
    
    print(f"[INFO] Teacher policy loaded successfully")
    print(f"  - Observations: 41D (privileged state)")
    print(f"  - Actions: 8D")
    print(f"  - Architecture: [256, 128, 64]")
    print(f"  - Loaded {len(actor_state_dict)} parameters")
    
    return teacher


def create_student_policy(device: str) -> VisionStudentPolicy:
    """Create vision-based student policy."""
    print(f"\n[INFO] Creating vision-based student policy")
    
    # Student config
    proprio_dim = 24  # Joint states (8) + joint vels (8) + actions (8)
    action_dim = 8
    vision_feature_dim = 256
    hidden_dims = [512, 256, 128]
    
    student = VisionStudentPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        vision_feature_dim=vision_feature_dim,
        hidden_dims=hidden_dims,
        activation="elu",
    )
    
    student.to(device)
    
    print(f"[INFO] Student policy created")
    print(f"  - Proprioception: {proprio_dim}D")
    print(f"  - Vision: RGBD (360x640x4) -> {vision_feature_dim}D")
    print(f"  - Actions: {action_dim}D")
    print(f"  - Architecture: {hidden_dims}")
    
    return student


def train_distillation(
    env,
    student: VisionStudentPolicy,
    teacher: nn.Module,
    args,
):
    """
    Train student policy using Online DAgger.
    
    Online DAgger algorithm:
    1. Collect rollouts using student policy (or mixed student/teacher)
    2. Label all states with teacher actions
    3. Train student to match teacher actions via supervised learning
    4. Repeat
    """
    
    # Setup
    device = args.device
    num_envs = env.unwrapped.num_envs
    num_steps_per_env = 24  # Match teacher training
    batch_size = num_envs * num_steps_per_env
    
    # Optimizer
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Logging
    log_dir = Path(f"logs/rsl_rl/franka_droid_distillation/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Logging to: {log_dir}")
    
    # Training loop
    print(f"\n[INFO] Starting Online DAgger training")
    print(f"  - Max iterations: {args.max_iterations}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Steps per env: {num_steps_per_env}")
    
    for iteration in range(args.max_iterations):
        # Reset environment
        obs_dict, _ = env.reset()
        
        # Storage for training data (observations, not actions)
        proprio_list = []
        rgbd_list = []
        teacher_obs_list = []
        epoch_rewards = []
        
        # Collect rollout using student policy
        student.eval()  # Eval mode for data collection
        for step in range(num_steps_per_env):
            # Get observations
            proprio = obs_dict["policy"]  # (num_envs, 24)
            teacher_obs = obs_dict["teacher_obs"]  # (num_envs, 41)
            rgbd = obs_dict["wrist_cam_rgb"]  # (num_envs, 360, 640, 3)
            depth = obs_dict["wrist_cam_depth"]  # (num_envs, 360, 640, 1)
            
            # Concatenate RGB + Depth
            rgbd_input = torch.cat([rgbd, depth], dim=-1)  # (num_envs, 360, 640, 4)
            
            # Store observations for training
            proprio_list.append(proprio.clone())
            rgbd_list.append(rgbd_input.clone())
            teacher_obs_list.append(teacher_obs.clone())
            
            # Get student action for rollout
            with torch.no_grad():
                student_action = student.act(proprio, rgbd_input, deterministic=False)
            
            # Step environment
            obs_dict, rewards, dones, infos = env.step(student_action)
            epoch_rewards.append(rewards.mean().item())
        
        # Train student on collected observations
        student.train()
        
        # Stack all observations
        all_proprio = torch.stack(proprio_list, dim=0)  # (steps, num_envs, 24)
        all_rgbd = torch.stack(rgbd_list, dim=0)  # (steps, num_envs, 360, 640, 4)
        all_teacher_obs = torch.stack(teacher_obs_list, dim=0)  # (steps, num_envs, 41)
        
        # Flatten batch: (steps * num_envs, ...)
        batch_proprio = all_proprio.view(-1, 24)
        batch_rgbd = all_rgbd.view(-1, 360, 640, 4)
        batch_teacher_obs = all_teacher_obs.view(-1, 41)
        
        # Get teacher actions (labels) - no grad needed
        with torch.no_grad():
            teacher_actions = teacher.act(batch_teacher_obs.to(device), deterministic=True)
        
        # Get student actions (with grad for backprop)
        student_actions = student.act(batch_proprio, batch_rgbd, deterministic=False)
        
        # Compute loss
        loss = criterion(student_actions, teacher_actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Logging
        mean_reward = sum(epoch_rewards) / len(epoch_rewards)
        print(f"[Iter {iteration:4d}] Loss: {loss.item():.6f} | Reward: {mean_reward:.3f}")
        
        # Save checkpoint
        if iteration % 50 == 0 and iteration > 0:
            checkpoint_path = log_dir / f"model_{iteration}.pt"
            torch.save({
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": iteration,
            }, checkpoint_path)
            print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    
    # Final save
    final_path = log_dir / "model_final.pt"
    torch.save({
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": args.max_iterations,
    }, final_path)
    print(f"\n[INFO] Training complete! Final model saved: {final_path}")
    
    return student


def main():
    """Main training function."""
    
    print("\n" + "="*80)
    print("VISION DISTILLATION TRAINING - DETAILED LOG")
    print("="*80)
    
    # Create environment
    print(f"\n[STEP 1/5] Creating environment: {args_cli.task}")
    print(f"  - Num envs: {args_cli.num_envs}")
    print(f"  - Device: {args_cli.device}")
    
    from isaaclab_tasks.direct.franka_droid.franka_droid_distillation_env import FrankaDroidDistillationEnvCfg
    
    env_cfg = FrankaDroidDistillationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    print(f"  - Config created, initializing environment...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print(f"  ✓ Environment created successfully!")
    
    # Wrap environment
    print(f"\n[STEP 2/5] Wrapping environment...")
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)  # Match RSL-RL config
    print(f"  ✓ Environment wrapped")
    
    # Load teacher
    print(f"\n[STEP 3/5] Loading teacher policy...")
    teacher_path = f"logs/rsl_rl/franka_droid_direct/{args_cli.load_run}/{args_cli.checkpoint}"
    print(f"  - Path: {teacher_path}")
    teacher = load_teacher_policy(teacher_path, args_cli.device)
    print(f"  ✓ Teacher loaded")
    
    # Create student
    print(f"\n[STEP 4/5] Creating student policy...")
    student = create_student_policy(args_cli.device)
    print(f"  ✓ Student created")
    
    # Train
    print(f"\n[STEP 5/5] Starting distillation training...")
    print(f"  - Max iterations: {args_cli.max_iterations}")
    print("="*80 + "\n")
    
    trained_student = train_distillation(env, student, teacher, args_cli)
    
    # Cleanup
    print("\n" + "="*80)
    print("TRAINING COMPLETE - CLEANING UP")
    print("="*80)
    env.close()
    print("\n[INFO] Distillation complete!")


if __name__ == "__main__":
    main()
    simulation_app.close()

