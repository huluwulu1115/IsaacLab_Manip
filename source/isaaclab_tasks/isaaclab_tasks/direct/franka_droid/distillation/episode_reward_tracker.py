"""
Episode Reward Tracker - Unified episode reward tracking module

Used to uniformly track episode cumulative rewards during training and evaluation,
ensuring metrics are comparable.
"""

import torch


class EpisodeRewardTracker:
    """
    Utility class for tracking episode cumulative rewards.
    
    Used for:
    - Training: Track cumulative rewards of completed episodes
    - Evaluation: Track cumulative rewards per episode
    - TensorBoard: Record episode-level statistics
    """
    
    def __init__(self, num_envs: int, device: str):
        """
        Initialize tracker.
        
        Args:
            num_envs: Number of parallel environments
            device: PyTorch device
        """
        self.num_envs = num_envs
        self.device = device
        
        # Current episode cumulative rewards
        self.current_episode_rewards = torch.zeros(num_envs, device=device)
        self.current_episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.int32)
        
        # Statistics of completed episodes
        self.completed_episode_rewards = []
        self.completed_episode_lengths = []
        self.completed_episode_successes = []  # NEW: Track success rate
        
        # Running stats for TensorBoard
        self.recent_episode_count = 100  # Keep last 100 episodes
    
    def update(self, step_rewards: torch.Tensor, dones: torch.Tensor, infos: dict = None):
        """
        Update reward tracking.
        
        Args:
            step_rewards: (num_envs,) - Rewards for current step
            dones: (num_envs,) - Which environments completed an episode
            infos: Optional dict containing additional episode info (e.g., success flags)
        """
        # Accumulate rewards
        self.current_episode_rewards += step_rewards
        self.current_episode_lengths += 1
        
        # Check completed episodes
        done_indices = dones.nonzero(as_tuple=False).flatten()
        
        if len(done_indices) > 0:
            for idx in done_indices:
                # Record completed episode
                ep_reward = self.current_episode_rewards[idx].item()
                ep_length = self.current_episode_lengths[idx].item()
                
                # Check if episode was successful (from infos if available)
                is_success = False
                if infos is not None and "successes" in infos:
                    is_success = infos["successes"][idx].item() > 0.5
                else:
                    # Fallback: high reward usually means success
                    is_success = ep_reward > 2000
                
                self.completed_episode_rewards.append(ep_reward)
                self.completed_episode_lengths.append(ep_length)
                self.completed_episode_successes.append(1.0 if is_success else 0.0)
                
                # Reset buffer for this environment
                self.current_episode_rewards[idx] = 0
                self.current_episode_lengths[idx] = 0
    
    def get_stats(self) -> dict:
        """
        Get episode statistics (for TensorBoard and printing).
        
        Returns:
            stats: Dictionary containing mean, std, min, max statistics
        """
        if len(self.completed_episode_rewards) == 0:
            return {
                "episode_reward_mean": 0.0,
                "episode_reward_std": 0.0,
                "episode_reward_min": 0.0,
                "episode_reward_max": 0.0,
                "episode_length_mean": 0.0,
                "episodes_completed": 0,
            }
        
        # Use recent episodes (avoid old data influence)
        recent_rewards = self.completed_episode_rewards[-self.recent_episode_count:]
        recent_lengths = self.completed_episode_lengths[-self.recent_episode_count:]
        recent_successes = self.completed_episode_successes[-self.recent_episode_count:]
        
        # Calculate reward per step (efficiency metric)
        reward_per_step = []
        for r, l in zip(recent_rewards, recent_lengths):
            if l > 0:
                reward_per_step.append(r / l)
        
        return {
            "episode_reward_mean": sum(recent_rewards) / len(recent_rewards),
            "episode_reward_std": torch.tensor(recent_rewards).std().item() if len(recent_rewards) > 1 else 0.0,
            "episode_reward_min": min(recent_rewards),
            "episode_reward_max": max(recent_rewards),
            "episode_length_mean": sum(recent_lengths) / len(recent_lengths),
            "episodes_completed": len(self.completed_episode_rewards),
            "success_rate": sum(recent_successes) / len(recent_successes) * 100 if len(recent_successes) > 0 else 0.0,
            "reward_per_step": sum(reward_per_step) / len(reward_per_step) if len(reward_per_step) > 0 else 0.0,  # NEW!
        }
    
    def get_success_rate(self, threshold: float = 10.0) -> float:
        """
        Calculate success rate.
        
        Args:
            threshold: Success threshold for episode reward
            
        Returns:
            success_rate: Success rate (0-100)
        """
        if len(self.completed_episode_rewards) == 0:
            return 0.0
        
        recent_rewards = self.completed_episode_rewards[-self.recent_episode_count:]
        success_count = sum(1 for r in recent_rewards if r > threshold)
        
        return success_count / len(recent_rewards) * 100

