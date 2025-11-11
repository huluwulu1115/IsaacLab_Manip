#!/usr/bin/env python
"""Evaluate trained vision student policy."""

import argparse
import torch
import torch.nn as nn
import gymnasium as gym

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-DROID-Distillation-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--device", type=str, default="cuda:0")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks
from isaaclab_tasks.direct.franka_droid.distillation import (
    VisionStudentPolicyResNet,
    FrankaDroidDistillationEnvCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    print("\n" + "="*80)
    print("VISION STUDENT POLICY EVALUATION")
    print("="*80)
    
    # Create environment
    env_cfg = FrankaDroidDistillationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.enable_visualization_markers = True
    
    print(f"\n[1/3] Creating environment...")
    print(f"  - Num envs: {args_cli.num_envs}")
    print(f"  - Early termination: {env_cfg.enable_early_termination}")
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)
    print("  ✓ Environment ready")
    
    # Load student
    print(f"\n[2/3] Loading student from: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=args_cli.device)
    
    student = VisionStudentPolicyResNet(
        proprio_dim=24, action_dim=8,
        vision_feature_dim=256,
        hidden_dims=[512, 256, 128],
        activation="elu",
        image_height=env_cfg.wrist_cam.height,
        image_width=env_cfg.wrist_cam.width,
        encoder_type="resnet18",
        use_pretrained=False,
    ).to(args_cli.device)
    
    student.load_state_dict(checkpoint["student_state_dict"])
    student.eval()
    print("  ✓ Student loaded")
    
    # Evaluate
    print(f"\n[3/3] Evaluating {args_cli.num_episodes} episodes...")
    print("="*80 + "\n")
    
    obs_dict, _ = env.reset()
    
    episode_rewards = []
    episode_lengths = []
    current_rewards = torch.zeros(args_cli.num_envs, device=args_cli.device)
    current_lengths = torch.zeros(args_cli.num_envs, device=args_cli.device, dtype=torch.int32)
    episodes_completed = 0
    
    step = 0
    while episodes_completed < args_cli.num_episodes:
        proprio = obs_dict["policy"]
        rgbd = torch.cat([obs_dict["wrist_cam_rgb"], obs_dict["wrist_cam_depth"]], dim=-1)
        
        with torch.no_grad():
            action = student.act(proprio, rgbd, deterministic=True)
        
        obs_dict, rewards, dones, infos = env.step(action)
        
        current_rewards += rewards
        current_lengths += 1
        
        done_indices = dones.nonzero(as_tuple=False).flatten()
        if len(done_indices) > 0:
            for idx in done_indices:
                if episodes_completed < args_cli.num_episodes:
                    ep_reward = current_rewards[idx].item()
                    ep_length = current_lengths[idx].item()
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    episodes_completed += 1
                    
                    if episodes_completed <= 10 or episodes_completed % 10 == 0:
                        print(f"  Episode {episodes_completed}/{args_cli.num_episodes}: "
                              f"Reward: {ep_reward:.2f}, Length: {ep_length}")
                
                current_rewards[idx] = 0
                current_lengths[idx] = 0
        
        step += 1
        if step > args_cli.num_episodes * 500:
            break
    
    # Results
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    mean_length = sum(episode_lengths) / len(episode_lengths)
    
    # Success rate
    success_count = sum(1 for r, l in zip(episode_rewards, episode_lengths) if (r/l if l > 0 else 0) > 6.0)
    success_rate = success_count / len(episode_rewards) * 100
    
    # R/step
    r_per_steps = [r/l for r, l in zip(episode_rewards, episode_lengths) if l > 0]
    mean_r_per_step = sum(r_per_steps) / len(r_per_steps)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Episodes: {args_cli.num_episodes}")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{args_cli.num_episodes})")
    print(f"Reward per Step: {mean_r_per_step:.2f}")
    print(f"Episode Reward Mean: {mean_reward:.2f}")
    print(f"Episode Length Mean: {mean_length:.1f}")
    print("="*80)
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

