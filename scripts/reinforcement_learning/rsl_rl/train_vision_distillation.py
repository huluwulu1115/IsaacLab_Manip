#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Vision-based Policy Distillation - DEXTRAH-aligned Implementation

Usage:
    python train_vision_distillation.py --task Isaac-DROID-Distillation-v0 \
        --num_envs 512 --max_steps 100000 --load_run 2025-10-23_19-43-40 \
        --checkpoint model_1499.pt --headless --enable_cameras
"""

import argparse
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train vision-based student policy via distillation (DEXTRAH-aligned)")
parser.add_argument("--task", type=str, default="Isaac-DROID-Distillation-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=str, default=42)
parser.add_argument("--max_steps", type=int, default=100_000, help="Total training steps (DEXTRAH: 100k)")
parser.add_argument("--load_run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default="model_1499.pt")
parser.add_argument("--video", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks
from isaaclab_tasks.direct.franka_droid.distillation import (
    VisionStudentPolicy,
    VisionStudentPolicyResNet,
    EpisodeRewardTracker,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def load_teacher_policy(checkpoint_path: str, teacher_cfg: dict, device: str) -> nn.Module:
    """Load teacher from checkpoint with config."""
    print(f"\n[INFO] Loading teacher policy from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    num_obs = teacher_cfg["num_observations"]
    num_actions = teacher_cfg["num_actions"]
    hidden_dims = teacher_cfg["actor_hidden_dims"]
    activation = teacher_cfg["activation"]

    print(f"[INFO] Teacher configuration:")
    print(f"  - Observations: {num_obs}D")
    print(f"  - Actions: {num_actions}D")
    print(f"  - Hidden dims: {hidden_dims}")
    print(f"  - Activation: {activation}")

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
    if "std" in state_dict:
        actor_dict["std"] = state_dict["std"]
    teacher.load_state_dict(actor_dict, strict=False)
    teacher.eval()

    print(f"[INFO] Teacher loaded: {len(actor_dict)} parameters")
    return teacher


def main():
    """Main training function - DEXTRAH-aligned single loop."""

    print("\n" + "=" * 80)
    print("VISION DISTILLATION TRAINING - DEXTRAH-ALIGNED")
    print("=" * 80)

    # Setup
    from isaaclab_tasks.direct.franka_droid.distillation import (
        FrankaDroidDistillationEnvCfg,
        FrankaDroidDistillationRunnerCfg,
    )

    env_cfg = FrankaDroidDistillationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    distill_cfg = FrankaDroidDistillationRunnerCfg()

    # Get beta schedule parameters from config
    if distill_cfg.beta_strategy == "linear":
        beta_start = distill_cfg.beta_start_decay
        beta_end = distill_cfg.beta_end_decay
        use_linear_beta = True
        print(f"  - Beta schedule: Linear decay ({beta_start:,} → {beta_end:,} steps)")
        print("  - Using extended teacher support: floor beta after decay")
    else:  # "step" - DEXTRAH original
        beta_warmup_steps = 15_000  # DEXTRAH default
        use_linear_beta = False
        print(f"  - Beta schedule: Step function (warmup: {beta_warmup_steps:,} steps)")

    print(f"\n[1/4] Creating environment: {args_cli.num_envs} envs...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)
    print("  ✓ Environment ready")

    print(f"\n[2/4] Loading teacher...")
    teacher_path = f"logs/rsl_rl/franka_droid_direct/{args_cli.load_run}/{args_cli.checkpoint}"
    teacher = load_teacher_policy(teacher_path, distill_cfg.teacher_config, args_cli.device)

    print(f"\n[3/4] Creating student...")
    # Get camera resolution from environment config

    # Select encoder type
    USE_RESNET = True  # Set to True to use ResNet18 (DEXTRAH-style), False for SimpleCNN

    if USE_RESNET:
        student = VisionStudentPolicyResNet(
            proprio_dim=24,
            action_dim=8,
            vision_feature_dim=256,
            hidden_dims=[512, 256, 128],
            activation="elu",
            image_height=env_cfg.wrist_cam.height,
            image_width=env_cfg.wrist_cam.width,
            encoder_type="resnet18",  # "resnet18" or "dextrah" (with Transformer)
            use_pretrained=True,  # Use ImageNet pretrained weights
        ).to(args_cli.device)
    else:
        student = VisionStudentPolicy(
            proprio_dim=24,
            action_dim=8,
            vision_feature_dim=256,
            hidden_dims=[512, 256, 128],
            activation="elu",
            image_height=env_cfg.wrist_cam.height,
            image_width=env_cfg.wrist_cam.width,
        ).to(args_cli.device)

    print(f"  ✓ Student ready ({sum(p.numel() for p in student.parameters()):,} params)")
    print(f"    Camera resolution: {env_cfg.wrist_cam.height}×{env_cfg.wrist_cam.width}")

    # Optimizer & logging
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    criterion = nn.MSELoss(reduction="none")

    log_dir = Path(f"logs/rsl_rl/franka_droid_distillation/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[4/4] Starting distillation training...")
    print(f"  - Total steps: {args_cli.max_steps:,}")
    print(f"  - Log dir: {log_dir}")
    print("=" * 80 + "\n")

    # Initialize
    obs_dict, _ = env.reset()
    num_envs = args_cli.num_envs
    device = args_cli.device

    # === Episode Reward Tracker (unified reward tracking) ===
    reward_tracker = EpisodeRewardTracker(num_envs, device)

    # === SINGLE LOOP (DEXTRAH-style) ===
    step = 0
    epoch_losses = []
    epoch_step_rewards = []  # Step rewards (for reference only)

    log_interval = 10  # Log every 10 steps (like DEXTRAH)
    save_interval = 5000  # Save every 5k steps (like DEXTRAH)

    while step < args_cli.max_steps:
        # === Beta Scheduling ===
        if use_linear_beta:
            # Linear Decay (smoother transition)
            if step < beta_start:
                beta = 1.0
            elif step < beta_end:
                beta = 1.0 - (step - beta_start) / (beta_end - beta_start)
            else:
                beta = 0.0
        else:
            # Step Function (DEXTRAH original)
            beta = 1.0 if step < beta_warmup_steps else 0.0

        # Get observations
        proprio = obs_dict["policy"]
        teacher_obs = obs_dict["teacher_obs"]
        rgbd = torch.cat([obs_dict["wrist_cam_rgb"], obs_dict["wrist_cam_depth"]], dim=-1)

        # Get teacher action and distribution
        with torch.no_grad():
            teacher_action = teacher.act(teacher_obs.to(device))
            # Get teacher's sigma (fixed parameter)
            teacher_sigma = teacher.std  # Shape: (action_dim,)

        # Get student action and distribution
        student.train()
        student_mean, student_sigma = student.forward(proprio, rgbd)

        deterministic_phase = step >= distill_cfg.switch_to_deterministic_step
        if deterministic_phase:
            with torch.no_grad():
                student.log_action_std.data.fill_(distill_cfg.post_switch_log_std)
            student_sigma = torch.exp(student.log_action_std).unsqueeze(0).expand_as(student_mean)
            student_action = student_mean
        else:
            student_action = student_mean + student_sigma * torch.randn_like(student_mean)

        # Action mixing based on beta (DEXTRAH-style)
        use_teacher_mask = (torch.rand(num_envs, device=device) < beta).unsqueeze(1)
        stepping_action = torch.where(use_teacher_mask, teacher_action, student_action)

        eps = 1e-8
        weights = (1.0 / teacher_sigma) ** 2

        # Loss 1: Weighted L2 norm on action means (sqrt of weighted squared error)
        mean_diff = (student_mean - teacher_action) ** 2 * weights
        weighted_mean_loss = torch.sqrt(mean_diff.sum(dim=-1) + eps).mean()

        # Loss 2: L2 norm on action sigmas (per-environment)
        teacher_sigma_exp = teacher_sigma.unsqueeze(0).expand_as(student_sigma)
        sigma_diff = (student_sigma - teacher_sigma_exp) ** 2
        sigma_loss = torch.sqrt(sigma_diff.sum(dim=-1) + eps).mean()

        # Total loss
        loss = weighted_mean_loss + sigma_loss

        # Per-step update (DEXTRAH-style)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        # Step environment
        obs_dict, rewards, dones, infos = env.step(stepping_action.detach())

        # Update episode reward tracker (unified tracking)
        reward_tracker.update(rewards, dones, infos)

        # Logging
        epoch_losses.append(loss.item())
        epoch_step_rewards.append(rewards.mean().item())  # Step reward (for reference only)
        step += 1

        # Print log (every 10 steps like DEXTRAH)
        if step % log_interval == 0:
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            mean_step_reward = sum(epoch_step_rewards) / len(epoch_step_rewards)

            # Get episode stats (unified episode reward)
            ep_stats = reward_tracker.get_stats()

            # Monitor current action std
            current_std = student_sigma.mean().item()

            if ep_stats["episodes_completed"] > 0:
                print(
                    f"Step {step:6d} | Loss: {mean_loss:.4f} | "
                    f"Success: {ep_stats['success_rate']:.1f}% | "
                    f"R/step: {ep_stats['reward_per_step']:.1f} | "
                    f"Len: {ep_stats['episode_length_mean']:.0f} | "
                    f"Beta: {beta:.2f} | Std: {current_std:.2f}"
                )
            else:
                print(f"Step {step:6d} | Loss: {mean_loss:.4f} | Beta: {beta:.2f} | Std: {current_std:.3f}")

            epoch_losses = []
            epoch_step_rewards = []

        # Save checkpoint (every 5k steps like DEXTRAH)
        if step % save_interval == 0:
            ckpt_path = log_dir / f"model_step_{step}.pt"
            torch.save(
                {
                    "student_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                },
                ckpt_path,
            )
            print(f"  → Saved: {ckpt_path.name}")

    # Final save
    final_path = log_dir / "model_final.pt"
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        final_path,
    )

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print(f"  Final model: {final_path}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
