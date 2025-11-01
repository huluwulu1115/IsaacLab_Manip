#!/usr/bin/env python
"""
Evaluate vision-based student policy from distillation.

This script loads a trained student policy and evaluates it in the environment.

Usage:
    python play_vision_policy.py --checkpoint logs/.../model_final.pt --num_envs 16
"""

import argparse
import torch
import gymnasium as gym
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate vision-based student policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to student checkpoint")
parser.add_argument("--task", type=str, default="Isaac-DROID-Distillation-v0")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments for evaluation")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to evaluate")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks
from isaaclab_tasks.direct.franka_droid.distillation import (
    VisionStudentPolicy,
    VisionStudentPolicyResNet
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def load_student_policy(checkpoint_path: str, device: str, image_height: int, image_width: int):
    """Load trained student policy."""
    print(f"\n[INFO] Loading student policy from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check checkpoint parameters to determine which encoder to use
    state_dict_keys = list(checkpoint["student_state_dict"].keys())
    use_resnet = any("resnet" in key or "backbone" in key for key in state_dict_keys)
    
    # Create student policy (match training architecture)
    if use_resnet:
        student = VisionStudentPolicyResNet(
            proprio_dim=24,
            action_dim=8,
            vision_feature_dim=256,
            hidden_dims=[512, 256, 128],
            activation="elu",
            image_height=image_height,
            image_width=image_width,
            encoder_type="resnet18",
            use_pretrained=False,  # No need for pretrained weights when loading
        ).to(device)
        print(f"[INFO] Loading ResNet-based student policy")
    else:
        student = VisionStudentPolicy(
            proprio_dim=24,
            action_dim=8,
            vision_feature_dim=256,
            hidden_dims=[512, 256, 128],
            activation="elu",
            image_height=image_height,
            image_width=image_width,
        ).to(device)
        print(f"[INFO] Loading SimpleCNN-based student policy")
    
    # Load weights
    student.load_state_dict(checkpoint["student_state_dict"])
    student.eval()
    
    print(f"[INFO] Student policy loaded successfully")
    print(f"  - Parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"  - Camera: {image_height}×{image_width}")
    
    return student


def evaluate_policy(env, student, num_episodes: int, device: str):
    """Evaluate student policy."""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING STUDENT POLICY")
    print(f"{'='*80}")
    print(f"Note: Reward = Episode cumulative total reward (sum of all step rewards)")
    print(f"{'='*80}\n")
    
    num_envs = env.unwrapped.num_envs
    episode_rewards = []
    episode_lengths = []
    
    # Reset
    obs_dict, _ = env.reset()
    current_rewards = torch.zeros(num_envs, device=device)
    current_lengths = torch.zeros(num_envs, device=device)
    episodes_completed = 0
    
    print(f"Running {num_episodes} episodes across {num_envs} parallel environments...\n")
    
    step = 0
    while episodes_completed < num_episodes:
        # Get observations
        proprio = obs_dict["policy"]
        rgbd = torch.cat([obs_dict["wrist_cam_rgb"], obs_dict["wrist_cam_depth"]], dim=-1)
        
        # Get action from student (deterministic for evaluation)
        with torch.no_grad():
            action = student.act(proprio, rgbd, deterministic=True)
        
        # Step environment
        obs_dict, rewards, dones, infos = env.step(action)
        
        # Accumulate rewards
        current_rewards += rewards
        current_lengths += 1
        
        # Check for done episodes
        done_indices = dones.nonzero(as_tuple=False).flatten()
        if len(done_indices) > 0:
            for idx in done_indices:
                if episodes_completed < num_episodes:
                    ep_reward = current_rewards[idx].item()
                    ep_length = current_lengths[idx].item()
                    
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    episodes_completed += 1
                    
                    # Print every episode for detailed analysis
                    print(f"  Episode {episodes_completed}/{num_episodes} | "
                          f"EpisodeReward: {ep_reward:.2f} | Length: {ep_length:.0f}")
                
                # Reset this environment
                current_rewards[idx] = 0
                current_lengths[idx] = 0
        
        step += 1
        
        # Safety: prevent infinite loop
        if step > num_episodes * 500:
            print(f"\n[WARNING] Max steps reached, stopping evaluation")
            break
    
    # Compute statistics
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    std_reward = torch.tensor(episode_rewards).std().item()
    mean_length = sum(episode_lengths) / len(episode_lengths)
    
    # Calculate success rate (reward > 2000 usually means success)
    num_success = sum(1 for r in episode_rewards if r > 2000)
    success_rate = num_success / len(episode_rewards) * 100
    
    # Calculate reward per step (efficiency metric)
    reward_per_step = [r / l for r, l in zip(episode_rewards, episode_lengths) if l > 0]
    mean_reward_per_step = sum(reward_per_step) / len(reward_per_step) if len(reward_per_step) > 0 else 0.0
    
    # Print results (unified naming: Episode Reward = cumulative total reward per episode)
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {success_rate:.1f}% ({num_success}/{num_episodes})")  # Primary metric
    print(f"Reward per Step: {mean_reward_per_step:.2f}")  # NEW: Efficiency metric
    print(f"Episode Reward Mean: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Episode Reward Median: {sorted(episode_rewards)[len(episode_rewards)//2]:.2f}")
    print(f"Episode Reward Min: {min(episode_rewards):.2f}")
    print(f"Episode Reward Max: {max(episode_rewards):.2f}")
    print(f"Episode Length Mean: {mean_length:.1f}")
    print(f"{'='*80}")
    
    # Print reward distribution
    print(f"\nReward Distribution:")
    import numpy as np
    bins = np.linspace(min(episode_rewards), max(episode_rewards), 5)
    hist, _ = np.histogram(episode_rewards, bins=bins)
    for i in range(len(hist)):
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]} episodes")
    print(f"{'='*80}\n")
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "all_rewards": episode_rewards,
    }


def main():
    """Main evaluation function."""
    
    print("\n" + "="*80)
    print("VISION STUDENT POLICY EVALUATION")
    print("="*80)
    
    # Create environment
    from isaaclab_tasks.direct.franka_droid.distillation import FrankaDroidDistillationEnvCfg
    
    env_cfg = FrankaDroidDistillationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.enable_visualization_markers = True  # Enable for better visualization
    
    print(f"\n[1/3] Creating evaluation environment...")
    print(f"  - Num envs: {args_cli.num_envs}")
    print(f"  - Device: {args_cli.device}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)
    print("  ✓ Environment ready")
    
    # Load student policy
    print(f"\n[2/3] Loading student policy...")
    student = load_student_policy(
        args_cli.checkpoint,
        args_cli.device,
        env_cfg.wrist_cam.height,
        env_cfg.wrist_cam.width
    )
    
    # Evaluate
    print(f"\n[3/3] Starting evaluation...")
    results = evaluate_policy(env, student, args_cli.num_episodes, args_cli.device)
    
    # Save results
    results_path = Path(args_cli.checkpoint).parent / "evaluation_results.txt"
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Checkpoint: {args_cli.checkpoint}\n")
        f.write(f"Episodes: {args_cli.num_episodes}\n")
        f.write(f"Episode Reward (Mean): {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
        f.write(f"Episode Length (Mean): {results['mean_length']:.1f}\n")
    
    print(f"\n[INFO] Results saved to: {results_path}")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

