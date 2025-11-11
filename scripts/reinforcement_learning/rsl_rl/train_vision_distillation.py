#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Vision-based Policy Distillation - DEXTRAH-aligned Implementation with improvements

Usage:
    python train_vision_distillation.py --task Isaac-DROID-Distillation-v0 \
        --num_envs 128 --max_steps 100000 --load_run 2025-10-23_19-43-40 \
        --checkpoint model_1499.pt --headless --enable_cameras
"""

import argparse
import os
import re
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
import glob
import wandb
import numpy as np
from copy import deepcopy

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train vision-based student policy via distillation")
parser.add_argument("--task", type=str, default="Isaac-DROID-Distillation-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=100_000)
parser.add_argument("--load_run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default="model_1499.pt")
parser.add_argument("--early_termination", action="store_true", default=False,
                    help="Enable early termination when success detected (pass flag to enable)")
parser.add_argument("--beta_start_decay", type=int, default=None,
                    help="Step to start beta decay (default: use config value)")
parser.add_argument("--beta_end_decay", type=int, default=None,
                    help="Step to end beta decay (default: use config value)")
parser.add_argument("--use_kl_loss", action="store_true", default=False,
                    help="Use KL divergence loss instead of DEXTRAH weighted MSE")
parser.add_argument("--target_std", type=float, default=0.3,
                    help="Target std for student to learn (default: 0.3, overrides teacher's high exploration std)")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record training videos (will slow down training)")
parser.add_argument("--video_interval", type=int, default=10000,
                    help="Interval between video recordings (in steps)")
parser.add_argument("--video_length", type=int, default=500,
                    help="Length of each recorded video (in steps)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks
from isaaclab_tasks.direct.franka_droid.distillation import (
    VisionStudentPolicyResNet,
    FrankaDroidDistillationRunnerCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper



def load_teacher_policy(checkpoint_path: str, teacher_cfg: dict, device: str) -> nn.Module:
    """Load teacher from checkpoint."""
    print(f"\n[INFO] Loading teacher policy from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    # Debug: Print all keys related to std
    print("\n[DEBUG] Checking std in checkpoint:")
    for key in state_dict.keys():
        if 'std' in key.lower():
            print(f"  Key: {key}, Shape: {state_dict[key].shape}, Value sample: {state_dict[key][:3] if len(state_dict[key].shape) > 0 else state_dict[key]}")
    
    num_obs = teacher_cfg["num_observations"]
    num_actions = teacher_cfg["num_actions"]
    hidden_dims = teacher_cfg["actor_hidden_dims"]
    activation = teacher_cfg["activation"]
    
    class TeacherActor(nn.Module):
        def __init__(self):
            super().__init__()
            act_fn = nn.ELU if activation == "elu" else nn.ReLU
            layers = []
            prev_dim = num_obs
            for h_dim in hidden_dims:
                layers.extend([nn.Linear(prev_dim, h_dim), act_fn()])
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, num_actions))
            self.network = nn.Sequential(*layers)
            self.std = nn.Parameter(torch.ones(num_actions), requires_grad=False)
        
        def act(self, obs, deterministic=True):
            return self.network(obs)
    
    teacher = TeacherActor().to(device)
    actor_dict = {k.replace("actor.", "network."): v for k, v in state_dict.items() if k.startswith("actor.")}
    
    # Load std - check for both "std" and "log_std"
    std_loaded = False
    if "std" in state_dict:
        actor_dict["std"] = state_dict["std"]
        std_loaded = True
        print(f"[DEBUG] Loaded 'std' directly: {state_dict['std']}")
    elif "log_std" in state_dict:
        # RSL-RL often saves log_std instead of std
        actor_dict["std"] = torch.exp(state_dict["log_std"])
        std_loaded = True
        print(f"[DEBUG] Loaded 'log_std' and converted: log_std={state_dict['log_std']}, std={torch.exp(state_dict['log_std'])}")
    elif "actor.std" in state_dict:
        actor_dict["std"] = state_dict["actor.std"]
        std_loaded = True
        print(f"[DEBUG] Loaded 'actor.std': {state_dict['actor.std']}")
    elif "actor.log_std" in state_dict:
        actor_dict["std"] = torch.exp(state_dict["actor.log_std"])
        std_loaded = True
        print(f"[DEBUG] Loaded 'actor.log_std' and converted: {torch.exp(state_dict['actor.log_std'])}")
    
    if not std_loaded:
        print("[WARNING] No std found in checkpoint, using default std=1.0")
    
    teacher.load_state_dict(actor_dict, strict=False)
    teacher.eval()
    
    print(f"\n[INFO] Teacher loaded:")
    print(f"  - Checkpoint std mean: {teacher.std.mean().item():.3f}")
    print(f"  - Checkpoint std per dim: {teacher.std.cpu().numpy()}")
    return teacher


def evaluate_on_training_env(
    student: nn.Module,
    env,
    device: str,
    num_episodes: int,
    step: int,
    teacher: nn.Module = None,
    video_dir: Path = None
):
    """Evaluate student policy on the current training environment (no new env creation).
    
    Args:
        student: Student policy network
        env: Current training environment (reused)
        device: Device to run evaluation on
        num_episodes: Number of episodes to evaluate
        step: Current training step
        teacher: Teacher policy (if provided, used at step 0 for baseline)
        video_dir: Video directory for wandb upload
    """
    # Use teacher at step 0 for baseline
    use_teacher_baseline = (step == 0 and teacher is not None)
    policy_name = "Teacher (Baseline)" if use_teacher_baseline else "Student"
    
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION @ Step {step:,} - {policy_name}")
    print(f"{'='*80}")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Mode: Deterministic (no exploration noise)")
    if use_teacher_baseline:
        print(f"  - Using: Teacher policy (state-based, as baseline)")
    else:
        print(f"  - Using: Student policy (vision-based)")
    print(f"  - Note: Using training env (with video recording if enabled)")
    
    # Get number of envs and reset for evaluation
    num_envs = env.unwrapped.num_envs
    obs_dict, _ = env.reset()  # Get initial observations
    
    episode_rewards = []
    episode_lengths = []
    current_rewards = torch.zeros(num_envs, device=device)
    current_lengths = torch.zeros(num_envs, device=device, dtype=torch.int32)
    episodes_completed = 0
    
    student.eval()
    eval_step = 0
    max_eval_steps = num_episodes * 500  # Safety limit
    
    with torch.no_grad():
        while episodes_completed < num_episodes and eval_step < max_eval_steps:
            # Get action from teacher or student
            if use_teacher_baseline:
                # Use teacher (state-based)
                teacher_obs = obs_dict["teacher_obs"]
                action = teacher.act(teacher_obs.to(device))
            else:
                # Use student (vision-based)
                proprio = obs_dict["policy"]
                rgbd = torch.cat([obs_dict["wrist_cam_rgb"], obs_dict["wrist_cam_depth"]], dim=-1)
                action = student.act(proprio, rgbd, deterministic=True)
            
            obs_dict, rewards, dones, infos = env.step(action)
            
            current_rewards += rewards
            current_lengths += 1
            eval_step += 1
            
            done_indices = dones.nonzero(as_tuple=False).flatten()
            if len(done_indices) > 0:
                for idx in done_indices:
                    if episodes_completed < num_episodes:
                        ep_reward = current_rewards[idx].item()
                        ep_length = current_lengths[idx].item()
                        episode_rewards.append(ep_reward)
                        episode_lengths.append(ep_length)
                        episodes_completed += 1
                        
                        if episodes_completed % 10 == 0:
                            print(f"  Episode {episodes_completed}/{num_episodes}: "
                                  f"R={ep_reward:.2f}, Len={ep_length}")
                    
                    current_rewards[idx] = 0
                    current_lengths[idx] = 0
    
    # Compute statistics
    r_per_steps = [r/l for r, l in zip(episode_rewards, episode_lengths) if l > 0]
    
    # Success metrics
    success_count = sum(1 for r_per_step in r_per_steps if r_per_step > 6.0)
    success_rate = (success_count / len(r_per_steps) * 100) if r_per_steps else 0
    high_quality_count = sum(1 for r_per_step in r_per_steps if r_per_step > 8.0)
    high_quality_rate = (high_quality_count / len(r_per_steps) * 100) if r_per_steps else 0
    
    # R/step statistics
    mean_r_per_step = np.mean(r_per_steps) if r_per_steps else 0
    median_r_per_step = np.median(r_per_steps) if r_per_steps else 0
    max_r_per_step = np.max(r_per_steps) if r_per_steps else 0
    min_r_per_step = np.min(r_per_steps) if r_per_steps else 0
    
    # Length statistics
    mean_length = np.mean(episode_lengths) if episode_lengths else 0
    median_length = np.median(episode_lengths) if episode_lengths else 0
    
    # Success-only statistics
    success_lengths = [l for l, r_per_s in zip(episode_lengths, r_per_steps) if r_per_s > 6.0]
    mean_success_length = np.mean(success_lengths) if success_lengths else 0
    
    # Total reward (for reference)
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    
    print(f"\n  ✓ Evaluation Results (deterministic, {len(episode_rewards)} episodes):")
    print(f"    Success Rate: {success_rate:.1f}% ({success_count}/{len(episode_rewards)})")
    print(f"    High Quality (R/step>8): {high_quality_rate:.1f}%")
    print(f"    R/step: mean={mean_r_per_step:.2f}, median={median_r_per_step:.2f}, max={max_r_per_step:.2f}")
    print(f"    Length: mean={mean_length:.1f}, success_only={mean_success_length:.1f}")
    print(f"    Total Reward (ref): {mean_reward:.2f}")
    
    # Log eval metrics (video upload is handled by caller)
    if use_teacher_baseline:
        # Teacher baseline: record to summary (constant reference) and one point at step=0
        metrics = {
            "teacher_baseline/success_rate": success_rate,
            "teacher_baseline/high_quality_rate": high_quality_rate,
            "teacher_baseline/reward_per_step_mean": mean_r_per_step,
            "teacher_baseline/reward_per_step_median": median_r_per_step,
            "teacher_baseline/length_mean": mean_length,
            "teacher_baseline/total_reward_mean": mean_reward,
        }
        # Log to time series at step=0 (single point for reference)
        wandb.log(metrics, step=step)
        # Also log to run summary as constant reference values
        wandb.run.summary.update(metrics)
        print(f"  ℹ️  Teacher baseline recorded to summary (constant reference)")
    else:
        # Student eval: regular time series logging
        wandb.log({
            # Success metrics
            f"eval/success_rate": success_rate,
            f"eval/high_quality_rate": high_quality_rate,
            
            # R/step distribution (核心指标)
            f"eval/reward_per_step_mean": mean_r_per_step,
            f"eval/reward_per_step_median": median_r_per_step,
            f"eval/reward_per_step_max": max_r_per_step,
            f"eval/reward_per_step_min": min_r_per_step,
            
            # Length distribution
            f"eval/length_mean": mean_length,
            f"eval/length_median": median_length,
            f"eval/success_length_mean": mean_success_length,
            
            # Total reward (仅供参考)
            f"eval/total_reward_mean": mean_reward,
        }, step=step)
    
    print(f"{'='*80}\n")
    
    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "mean_r_per_step": mean_r_per_step,
    }


def main():
    """Main training function."""
    
    print("\n" + "="*80)
    print("VISION DISTILLATION TRAINING (RSL-RL)")
    print("="*80)
    
    # Setup
    from isaaclab_tasks.direct.franka_droid.distillation import FrankaDroidDistillationEnvCfg
    
    env_cfg = FrankaDroidDistillationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.enable_early_termination = args_cli.early_termination
    
    distill_cfg = FrankaDroidDistillationRunnerCfg()
    
    # Beta schedule (command line args override config)
    beta_start = args_cli.beta_start_decay if args_cli.beta_start_decay is not None else distill_cfg.beta_start_decay
    beta_end = args_cli.beta_end_decay if args_cli.beta_end_decay is not None else distill_cfg.beta_end_decay
    
    print(f"  - Beta schedule: {beta_start:,} → {beta_end:,} steps (linear decay, {beta_end-beta_start:,} steps transition)")
    print(f"  - Early termination: {'Enabled' if args_cli.early_termination else 'Disabled'}")
    print(f"  - Loss type: {'KL Divergence' if args_cli.use_kl_loss else 'DEXTRAH Weighted MSE'}")
    print(f"  - Target std: {args_cli.target_std} (overriding teacher's exploration std)")
    print(f"  - Teacher checkpoint std: will be displayed after loading")
    
    # Create log directory (needed for video recording)
    log_dir = Path(f"logs/rsl_rl/franka_droid_distillation/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/4] Creating environment: {args_cli.num_envs} envs...")
    print(f"  - Log dir: {log_dir}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap with video recorder if enabled (slows training ~30%)
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(log_dir / "videos" / "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"  [VIDEO] Recording enabled:")
        print(f"    - Interval: every {args_cli.video_interval:,} steps")
        print(f"    - Length: {args_cli.video_length} steps per video")
        print(f"    - Folder: {video_kwargs['video_folder']}")
        print(f"    ⚠️  Training will be ~30% slower")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)
    print("  ✓ Environment ready")
    
    print(f"\n[2/4] Loading teacher...")
    teacher_path = f"logs/rsl_rl/franka_droid_direct/{args_cli.load_run}/{args_cli.checkpoint}"
    teacher = load_teacher_policy(teacher_path, distill_cfg.teacher_config, args_cli.device)
    
    print(f"\n[3/4] Creating student...")
    student = VisionStudentPolicyResNet(
        proprio_dim=24, action_dim=8,
        vision_feature_dim=256,
        hidden_dims=[512, 256, 128],
        activation="elu",
        image_height=env_cfg.wrist_cam.height,
        image_width=env_cfg.wrist_cam.width,
        encoder_type="resnet18",
        use_pretrained=True,  # Use ImageNet pretrained
    ).to(args_cli.device)
    
    print(f"  ✓ Student ready ({sum(p.numel() for p in student.parameters()):,} params)")
    
    # Optimizer
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    
    # Wandb initialization (required)
    # Generate descriptive run name based on key parameters
    loss_type_str = "kl" if args_cli.use_kl_loss else "dextrah"
    et_str = "ET" if args_cli.early_termination else "noET"
    video_str = "vid" if args_cli.video else ""
    
    run_name = (
        f"{loss_type_str}_"
        f"b{beta_start//1000}k-{beta_end//1000}k_"
        f"std{args_cli.target_std:.1f}_"
        f"{et_str}"
    )
    if video_str:
        run_name += f"_{video_str}"
    run_name += f"_env{args_cli.num_envs}_{datetime.now().strftime('%m%d_%H%M')}"
    
    # Example: "dextrah_b15k-35k_std0.3_ET_env128_1111_1430"
    # With video: "dextrah_b15k-35k_std0.3_ET_vid_env128_1111_1430"
    
    wandb.init(
        project="franka_droid_distillation",
        name=run_name,
        config={
            "num_envs": args_cli.num_envs,
            "max_steps": args_cli.max_steps,
            "beta_start_decay": beta_start,
            "beta_end_decay": beta_end,
            "beta_transition_steps": beta_end - beta_start,
            "early_termination": args_cli.early_termination,
            "loss_type": "kl_divergence" if args_cli.use_kl_loss else "dextrah_weighted_mse",
            "target_std": args_cli.target_std,
            "video_enabled": args_cli.video,
            "video_interval": args_cli.video_interval if args_cli.video else None,
            "teacher_run": args_cli.load_run,
            "teacher_checkpoint": args_cli.checkpoint,
        }
    )
    print(f"  ✓ W&B initialized")
    
    print(f"\n[4/4] Starting distillation training...")
    print(f"  - Log dir: {log_dir}")
    print("="*80 + "\n")
    
    # Initialize
    obs_dict, _ = env.reset()
    num_envs = args_cli.num_envs
    device = args_cli.device
    
    # Episode tracking
    episode_rewards = []
    episode_lengths = []
    current_rewards = torch.zeros(num_envs, device=device)
    current_lengths = torch.zeros(num_envs, device=device, dtype=torch.int32)
    
    step = 0
    epoch_losses = []
    
    log_interval = 10
    save_interval = 5000
    eval_interval = 10000  # Evaluate every 10k steps
    num_eval_episodes = 50  # Number of episodes for evaluation (using training envs)
    
    # Video directory for finding recorded videos
    video_dir = log_dir / "videos" / "train" if args_cli.video else None
    
    # Initial evaluation (Step 0 baseline - use Teacher!)
    print(f"\n{'='*40}")
    print("  INITIAL EVALUATION (Teacher Baseline)")
    print(f"{'='*40}\n")
    
    initial_results = evaluate_on_training_env(
        student=student,
        env=env,
        device=device,
        num_episodes=num_eval_episodes,
        step=0,
        teacher=teacher,  # Pass teacher for step 0 baseline
        video_dir=video_dir,  # Record teacher baseline video
    )
    
    print(f"  Teacher baseline: Success={initial_results['success_rate']:.1f}%, R/step={initial_results['mean_r_per_step']:.2f}")
    print(f"  (This is the performance target for student)")
    if args_cli.video:
        print(f"  ℹ️  Teacher baseline video will be uploaded at step 10k\n")
    else:
        print()
    
    # Reset environment after eval
    obs_dict, _ = env.reset()
    current_rewards = torch.zeros(num_envs, device=device)
    current_lengths = torch.zeros(num_envs, device=device, dtype=torch.int32)
    
    # Track previous videos for delayed upload
    uploaded_videos = set()  # Keep track of what we've uploaded
    
    while step < args_cli.max_steps:
        # Beta scheduling
        if step < beta_start:
            beta = 1.0
        elif step < beta_end:
            beta = 1.0 - (step - beta_start) / (beta_end - beta_start)
        else:
            beta = 0.0
        
        # Get observations
        proprio = obs_dict["policy"]
        teacher_obs = obs_dict["teacher_obs"]
        rgbd = torch.cat([obs_dict["wrist_cam_rgb"], obs_dict["wrist_cam_depth"]], dim=-1)
        
        # Get teacher action and sigma
        with torch.no_grad():
            teacher_action = teacher.act(teacher_obs.to(device))
            # Override teacher's high exploration std with reasonable target
            # Teacher uses deterministic mean in eval, so student should have small std
            teacher_sigma = torch.ones_like(teacher.std) * args_cli.target_std
        
        # Get student action and distribution
        student.train()
        student_mean, student_sigma = student.forward(proprio, rgbd)
        student_action = student_mean + student_sigma * torch.randn_like(student_mean)
        
        # Action mixing
        use_teacher_mask = (torch.rand(num_envs, device=device) < beta).unsqueeze(1)
        stepping_action = torch.where(use_teacher_mask, teacher_action, student_action)
        
        # Loss computation (DEXTRAH vs KL)
        if args_cli.use_kl_loss:
            # KL Divergence: KL(Teacher || Student)
            # For Gaussian: KL(N(μ_t, σ_t) || N(μ_s, σ_s))
            kl_per_dim = (
                torch.log(student_sigma / (teacher_sigma + 1e-8))  # log ratio
                + (teacher_sigma ** 2 + (teacher_action - student_mean) ** 2) / (2 * student_sigma ** 2 + 1e-8)
                - 0.5
            )
            loss = kl_per_dim.mean()
        else:
            # DEXTRAH-style Loss (default)
            weights = (1.0 / teacher_sigma) ** 2
            weighted_mean_loss = (((student_mean - teacher_action) ** 2) * weights).mean()
            sigma_loss = ((student_sigma.mean(dim=0) - teacher_sigma) ** 2).mean()
            loss = weighted_mean_loss + sigma_loss
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step environment
        obs_dict, rewards, dones, infos = env.step(stepping_action.detach())
        
        # Track episodes
        current_rewards += rewards
        current_lengths += 1
        
        done_indices = dones.nonzero(as_tuple=False).flatten()
        if len(done_indices) > 0:
            for idx in done_indices:
                ep_reward = current_rewards[idx].item()
                ep_length = current_lengths[idx].item()
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                current_rewards[idx] = 0
                current_lengths[idx] = 0
        
        # Logging
        epoch_losses.append(loss.item())
        step += 1
        
        # Print log
        if step % log_interval == 0:
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            
            # Calculate stats from recent episodes
            if len(episode_rewards) >= 10:
                recent_rewards = episode_rewards[-100:]
                recent_lengths = episode_lengths[-100:]
                
                # Success rate (R/step > 6)
                r_per_steps = [r/l for r, l in zip(recent_rewards, recent_lengths) if l > 0]
                success_count = sum(1 for r_per_step in r_per_steps if r_per_step > 6.0)
                success_rate = (success_count / len(r_per_steps) * 100) if r_per_steps else 0
                
                # R/step statistics
                mean_r_per_step = sum(r_per_steps) / len(r_per_steps) if r_per_steps else 0
                median_r_per_step = sorted(r_per_steps)[len(r_per_steps)//2] if r_per_steps else 0
                max_r_per_step = max(r_per_steps) if r_per_steps else 0
                
                # Length statistics
                mean_length = sum(recent_lengths) / len(recent_lengths)
                min_length = min(recent_lengths) if recent_lengths else 0
                
                # Success-only statistics (episodes with R/step > 6.0)
                success_rewards = [r for r, r_per_s in zip(recent_rewards, r_per_steps) if r_per_s > 6.0]
                success_lengths = [l for l, r_per_s in zip(recent_lengths, r_per_steps) if r_per_s > 6.0]
                mean_success_length = (sum(success_lengths) / len(success_lengths)) if success_lengths else 0
                
                # High-quality success rate (R/step > 8.0 = fast completion)
                high_quality_count = sum(1 for r_per_step in r_per_steps if r_per_step > 8.0)
                high_quality_rate = (high_quality_count / len(r_per_steps) * 100) if r_per_steps else 0
                
                # Std
                current_std = student_sigma.mean().item()
                
                print(f"Step {step:6d} | Loss: {mean_loss:.4f} | "
                      f"Success: {success_rate:.1f}% | "
                      f"R/step: {mean_r_per_step:.1f} (max:{max_r_per_step:.1f}) | "
                      f"Len: {mean_length:.0f} | "
                      f"Beta: {beta:.2f} | Std: {current_std:.2f}")
                
                # Wandb
                wandb.log({
                    # Training metrics
                    "training/loss": mean_loss,
                    "training/beta": beta,
                    "training/std": current_std,
                    
                    # Success metrics
                    "episode/success_rate": success_rate,
                    "episode/high_quality_rate": high_quality_rate,  # R/step > 8.0
                    
                    # R/step distribution (核心指标)
                    "episode/reward_per_step_mean": mean_r_per_step,
                    "episode/reward_per_step_median": median_r_per_step,
                    "episode/reward_per_step_max": max_r_per_step,
                    
                    # Episode length distribution
                    "episode/length_mean": mean_length,
                    "episode/length_min": min_length,
                    "episode/success_length_mean": mean_success_length,  # 只统计成功的
                    
                    # Total reward (仅供参考)
                    "episode/total_reward_mean": sum(recent_rewards)/len(recent_rewards),
                }, step=step)
            else:
                print(f"Step {step:6d} | Loss: {mean_loss:.4f} | Beta: {beta:.2f}")
            
            epoch_losses = []
        
        # Save checkpoint
        if step % save_interval == 0 and step > 0:
            ckpt_path = log_dir / f"model_step_{step}.pt"
            torch.save({
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            }, ckpt_path)
            print(f"  → Saved: {ckpt_path.name}")
        
        # Periodic evaluation
        if step % eval_interval == 0 and step > 0:
            print(f"\n{'📊'*40}")
            print(f"  PERIODIC EVALUATION @ Step {step:,}")
            print(f"{'📊'*40}\n")
            
            eval_results = evaluate_on_training_env(
                student=student,
                env=env,
                device=device,
                num_episodes=num_eval_episodes,
                step=step,
                teacher=None,  # Use student for periodic evaluations
                video_dir=video_dir,
            )
            
            print(f"  Evaluation @ {step:,}: Success={eval_results['success_rate']:.1f}%, R/step={eval_results['mean_r_per_step']:.2f}\n")
            
            # Upload previous period's videos (already written and stable)
            if args_cli.video and video_dir and video_dir.exists():
                print(f"  📤 Uploading videos from previous periods...")
                all_videos = sorted(glob.glob(str(video_dir / "*.mp4")))
                new_videos = [v for v in all_videos if v not in uploaded_videos]
                
                if new_videos:
                    for video_path in new_videos:
                        video_name = os.path.basename(video_path)
                        print(f"     Uploading: {video_name} ({os.path.getsize(video_path)/1024:.1f} KB)")
                        
                        # Extract step from filename using regex (same as final upload)
                        match = re.search(r'step-(\d+)', video_name)
                        if match:
                            upload_step = int(match.group(1))
                        else:
                            # Fallback: estimate from number of uploaded videos
                            upload_step = len(uploaded_videos) * eval_interval
                            print(f"     ⚠️  Could not extract step from filename, using fallback: {upload_step}")
                        
                        # Determine key based on step
                        if upload_step == 0:
                            # Teacher baseline - upload under teacher_baseline section
                            wandb_key = "teacher_baseline/video"
                        else:
                            # Student progress - upload under eval section
                            wandb_key = "eval/video"
                        
                        try:
                            wandb.log({wandb_key: wandb.Video(video_path, fps=30, format="mp4")}, step=upload_step)
                            uploaded_videos.add(video_path)
                            print(f"     ✅ {video_name} uploaded to {wandb_key} at step {upload_step}")
                        except Exception as e:
                            print(f"     ❌ Failed: {e}")
                    
                    print(f"  ✓ Uploaded {len(new_videos)} video(s)\n")
                else:
                    print(f"  ℹ️  No new videos to upload\n")
            
            # Reset environment after eval
            obs_dict, _ = env.reset()
            current_rewards = torch.zeros(num_envs, device=device)
            current_lengths = torch.zeros(num_envs, device=device, dtype=torch.int32)
    
    # Final save
    final_path = log_dir / "model_final.pt"
    torch.save({
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }, final_path)
    
    # Final evaluation
    print(f"\n{'🎉'*40}")
    print("  FINAL EVALUATION")
    print(f"{'🎉'*40}\n")
    
    final_results = evaluate_on_training_env(
        student=student,
        env=env,
        device=device,
        num_episodes=num_eval_episodes * 2,  # More episodes for final eval
        step=step,
        teacher=None,  # Use student for final evaluation
        video_dir=video_dir,
    )
    
    # Upload all remaining videos at the end
    if args_cli.video and video_dir and video_dir.exists():
        import time
        import re
        
        print(f"\n{'📹'*40}")
        print("  WAITING FOR FINAL VIDEO")
        print(f"{'📹'*40}\n")
        
        # Wait for the last video to be fully written (max 10 minutes)
        print(f"  ⏳ Waiting for final video to be written (max 10 min)...")
        max_wait = 600  # 10 minutes
        wait_time = 0
        prev_size = 0
        stable_count = 0
        last_video_count = len(glob.glob(str(video_dir / "*.mp4")))
        
        while wait_time < max_wait:
            all_videos = sorted(glob.glob(str(video_dir / "*.mp4")), key=os.path.getmtime)
            current_video_count = len(all_videos)
            
            if all_videos:
                latest_video = all_videos[-1]
                current_size = os.path.getsize(latest_video)
                
                # Check if file size is stable (video writing complete)
                if current_size == prev_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= 3:  # 连续3次稳定（6秒）
                        print(f"  ✓ Final video ready: {os.path.basename(latest_video)} ({current_size/1024:.1f} KB)")
                        print(f"  ✓ Wait time: {wait_time}s\n")
                        break
                else:
                    stable_count = 0
                    prev_size = current_size
                    if wait_time % 10 == 0 and wait_time > 0:  # 每10秒更新一次
                        print(f"     Still writing... ({current_size/1024:.1f} KB, {wait_time}s elapsed)")
            
            time.sleep(2)
            wait_time += 2
        
        # Upload all videos
        print(f"\n  📤 Uploading all videos to wandb...")
        all_videos = sorted(glob.glob(str(video_dir / "*.mp4")))
        new_videos = [v for v in all_videos if v not in uploaded_videos]
        
        if new_videos:
            for video_path in new_videos:
                video_name = os.path.basename(video_path)
                
                # Extract step from filename
                match = re.search(r'step-(\d+)', video_name)
                if match:
                    video_step = int(match.group(1))
                else:
                    video_step = step
                    print(f"     ⚠️  Could not extract step from filename, using final step: {video_step}")
                
                # Determine key based on step (consistent with training uploads)
                if video_step == 0:
                    wandb_key = "teacher_baseline/video"
                else:
                    wandb_key = "eval/video"
                
                print(f"     Uploading: {video_name} ({os.path.getsize(video_path)/1024:.1f} KB) → {wandb_key} (step={video_step})")
                try:
                    wandb.log({wandb_key: wandb.Video(video_path, fps=30, format="mp4")}, step=video_step)
                    uploaded_videos.add(video_path)
                    print(f"     ✅ Uploaded")
                except Exception as e:
                    print(f"     ❌ Failed: {e}")
            
            print(f"\n  ✓ Total uploaded: {len(new_videos)} video(s)")
        else:
            print(f"  ℹ️  All videos already uploaded")
        
        print(f"{'='*80}\n")
    
    wandb.finish()
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print(f"  Final model: {final_path}")
    print(f"  Final Success Rate: {final_results['success_rate']:.1f}%")
    print(f"  Final R/step: {final_results['mean_r_per_step']:.2f}")
    print("="*80)
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

