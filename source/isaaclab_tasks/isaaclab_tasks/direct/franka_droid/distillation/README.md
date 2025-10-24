# Distillation Package

This package contains all components for vision-based policy distillation in the Franka DROID environment.

## Package Structure

```
distillation/
├── __init__.py                      # Package exports
├── distillation_env.py              # Environment configuration for distillation
├── rsl_rl_distillation_cfg.py      # RSL-RL distillation training configuration
├── vision_policy_network.py        # Vision-based student policy (SimpleCNN)
├── vision_policy_resnet.py          # Vision-based student policy (ResNet18)
├── resnet_encoder.py                # ResNet18 RGBD encoder
└── episode_reward_tracker.py       # Episode reward tracking utilities
```

## Quick Start

### Import the package

```python
from isaaclab_tasks.direct.franka_droid.distillation import (
    FrankaDroidDistillationEnv,
    FrankaDroidDistillationEnvCfg,
    FrankaDroidDistillationRunnerCfg,
    VisionStudentPolicy,
    VisionStudentPolicyResNet,
)
```

### Use the distillation environment

```python
# Register environment
import gymnasium as gym
env = gym.make("Isaac-DROID-Distillation-v0", num_envs=512)
```

## Components

### 1. Environment Configuration (`distillation_env.py`)
- **FrankaDroidDistillationEnvCfg**: Configuration for distillation with vision observations
- **FrankaDroidDistillationEnv**: Environment with RGBD camera support

### 2. Vision Encoders
- **SimpleCNNEncoder** (`vision_policy_network.py`): Lightweight CNN encoder for RGBD
- **ResNet18RGBDEncoder** (`resnet_encoder.py`): ResNet18-based encoder with pretrained weights
- **DexterahStyleEncoder** (`resnet_encoder.py`): Full DEXTRAH-style encoder (ResNet18 + Transformer)

### 3. Student Policies
- **VisionStudentPolicy** (`vision_policy_network.py`): Student policy with SimpleCNN encoder
- **VisionStudentPolicyResNet** (`vision_policy_resnet.py`): Student policy with ResNet18 encoder

### 4. Training Configuration (`rsl_rl_distillation_cfg.py`)
- **FrankaDroidDistillationRunnerCfg**: Complete training configuration
  - Online DAgger distillation
  - Beta scheduling (teacher → student transition)
  - DEXTRAH-aligned hyperparameters

### 5. Utilities
- **EpisodeRewardTracker** (`episode_reward_tracker.py`): Track episode-level rewards

## Training Scripts

Located in `scripts/reinforcement_learning/rsl_rl/`:
- `train_vision_distillation.py`: Train student policy via distillation
- `play_vision_policy.py`: Evaluate trained student policy

## Migration from Previous Structure

**Old imports:**
```python
from isaaclab_tasks.direct.franka_droid.agents.vision_policy_network import VisionStudentPolicy
from isaaclab_tasks.direct.franka_droid.franka_droid_distillation_env import FrankaDroidDistillationEnvCfg
```

**New imports:**
```python
from isaaclab_tasks.direct.franka_droid.distillation import (
    VisionStudentPolicy,
    FrankaDroidDistillationEnvCfg
)
```

## Features

- ✅ RGBD camera support (TiledCamera for multi-environment efficiency)
- ✅ Multiple encoder options (SimpleCNN, ResNet18, DEXTRAH-style)
- ✅ Online DAgger distillation with beta scheduling
- ✅ Episode-level reward tracking
- ✅ Teacher policy loading from checkpoint
- ✅ DEXTRAH-aligned implementation

## Related Files

- Main environment: `../franka_droid_env.py`
- Teacher configuration: `../agents/rsl_rl_ppo_cfg.py`
- Training scripts: `scripts/reinforcement_learning/rsl_rl/`

