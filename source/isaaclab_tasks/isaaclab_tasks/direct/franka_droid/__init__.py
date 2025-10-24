# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
DROID manipulation environment.

This environment provides a Franka Panda robot with Robotiq 2F-85 gripper
for manipulation tasks using the Direct RL workflow.

Available versions:
- v0: Original version with manual actuator configuration
- v1: Simplified version that reads physics parameters from USD
"""

import gymnasium as gym

from . import agents
from . import distillation

##
# Register Gym environments
##

# v0: Original version with manual configuration
gym.register(
    id="Isaac-DROID-Direct-v0",
    entry_point=f"{__name__}.franka_droid_env:FrankaDroidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_droid_env:FrankaDroidEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaDroidPPORunnerCfg",
    },
)

# v1: Simplified version reading from USD
gym.register(
    id="Isaac-DROID-Direct-v1",
    entry_point=f"{__name__}.franka_droid_env_v1:FrankaDroidEnv_v1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_droid_env_v1:FrankaDroidEnvCfg_v1",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaDroidPPORunnerCfg",
    },
)

# Distillation: Vision-based student policy (RGBD + proprioception)
# Distills teacher policy (state-based) to student policy (vision-based)
gym.register(
    id="Isaac-DROID-Distillation-v0",
    entry_point=f"{__name__}.distillation.distillation_env:FrankaDroidDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.distillation.distillation_env:FrankaDroidDistillationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{distillation.__name__}.rsl_rl_distillation_cfg:FrankaDroidDistillationRunnerCfg",
    },
)

