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
    
    # Disable visualization markers for training (saves GPU resources)
    enable_visualization_markers: bool = False
    
    # Reduce number of environments for memory efficiency (vision is memory-intensive)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=2.5, replicate_physics=True
    )
    
    # Rest of the configuration inherits from FrankaDroidEnvCfg


# Alias for easier import
FrankaDroidDistillationEnv = FrankaDroidEnv

