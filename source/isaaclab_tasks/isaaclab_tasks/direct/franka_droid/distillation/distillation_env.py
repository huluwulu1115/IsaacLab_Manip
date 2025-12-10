"""
Franka DROID Distillation Environment Configuration.

This configuration enables vision-based observations for student policy distillation.
"""

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from ..franka_droid_env import FrankaDroidEnv, FrankaDroidEnvCfg


@configclass
class FrankaDroidDistillationEnvCfg(FrankaDroidEnvCfg):
    """
    Environment configuration for vision-based distillation.

    Automatically enables:
    - Vision observations (RGBD camera)
    - Teacher observations (privileged state)
    - Smaller batch size for memory efficiency
    """

    # Enable vision observations for student
    use_vision_observations: bool = True

    # Enable early termination on success (critical for distillation!)
    # This prevents data imbalance: too much "fine-tuning at goal" data
    # Can be overridden by --early_termination command line argument
    enable_early_termination: bool = True
    early_termination_hold_time: float = 0.5  # Time to stay at goal before terminating (matches main env)

    # Disable visualization markers for training (saves GPU resources)
    enable_visualization_markers: bool = False

    # Reduce number of environments for memory efficiency (vision is memory-intensive)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=True)

    # Rest of the configuration inherits from FrankaDroidEnvCfg


# Alias for easier import
FrankaDroidDistillationEnv = FrankaDroidEnv
