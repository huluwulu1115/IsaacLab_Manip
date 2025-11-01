"""Script to perform student-teacher distillation"""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--teacher", type=str, default=None, help="Teacher checkpoint to use")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import math
import os
from datetime import datetime
import pathlib
import torch.distributed as dist

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


from distillation_transformer import Dagger
import dextrah_lab.tasks.dextrah_kuka_allegro.gym_setup

from dextrah_lab.distillation.a2c_with_aux_depth import A2CBuilder as A2CWithAuxDepthBuilder
from dextrah_lab.distillation.a2c_with_aux_cnn import A2CBuilder as A2CWithAuxCNNBuilder
from dextrah_lab.distillation.a2c_with_aux_cnn_stereo import A2CBuilder as A2CWithAuxCNNStereoBuilder
from dextrah_lab.distillation.a2c_with_aux_cnn_stereo_recon import A2CBuilder as A2CWithAuxCNNStereoReconBuilder
from dextrah_lab.distillation.a2c_with_aux_transformer_stereo import A2CBuilder as A2CWithAuxTransformerStereoBuilder
from dextrah_lab.distillation.a2c_with_aux_cnn_transformer_stereo_flow import A2CBuilder as A2CWithAuxTransformerStereoFlowBuilder


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """ Performs distillation. """
    world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
    rank = int(os.environ['RANK'])  # Global rank of this process
    local_rank = int(os.environ['LOCAL_RANK']) # local rank of the process 
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # parse configuration
    # env_cfg = parse_env_cfg(
    #     args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    # )
    # agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    ov_env = env.env

    parent_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
    agent_cfg_folder = "dextrah_lab/tasks/dextrah_kuka_allegro/agents"

    student_cfg = os.path.join(
        parent_path,
        agent_cfg_folder,
        "rl_games_ppo_transformer_stereo.yaml",
        # "rl_games_ppo_transformer_stereo_flow.yaml",
    )
    teacher_cfg = os.path.join(
        parent_path,
        agent_cfg_folder,
        "rl_games_ppo_lstm_cfg.yaml"
    )

    num_student_obs = ov_env.num_observations
    num_teacher_obs = ov_env.num_teacher_observations
    num_actions = ov_env.num_actions
    student_ckpt = "pretrained_ckpts/dextrah_student_30000_iters.pth"
    student_ckpt = os.path.join(
        parent_path,
        student_ckpt
    )
    student_ckpt='/home/ritviks/workspace/dextrah_distillation_results/dextrah_stereo_transformer_new_bounds_5/model/nn/dextrah_student_55000_iters.pth'
    student_ckpt = "/home/ritviks/workspace/git/dextrah_lab/pretrained_ckpts/dextrah_student_115000.pth"
    student_ckpt = None
    # student_ckpt = "/home/ritviks/workspace/git/dextrah_lab/dextrah_lab/distillation/runs/Dextrah-Kuka-Allegro_10-18-40-54/nn/dextrah_student_15000_iters.pth"
    if args_cli.teacher is not None:
        teacher_ckpt = os.path.join("pretrained_ckpts", args_cli.teacher)
    else:
        teacher_ckpt = "pretrained_ckpts/new_teacher.pth"
    teacher_ckpt = os.path.join(
        parent_path,
        teacher_ckpt
    )

    if rank == 0:
        train_dir = "runs"
        experiment_name = (
            "Dextrah-Kuka-Allegro"
            + datetime.now().strftime("_%d-%H-%M-%S")
        )
        experiment_dir = os.path.join(train_dir, experiment_name)
        nn_dir = os.path.join(experiment_dir, "nn")
        summaries_dir = os.path.join(experiment_dir, "summaries")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(nn_dir, exist_ok=True)
        os.makedirs(summaries_dir, exist_ok=True)
    else:
        summaries_dir = None
        nn_dir = None

    dagger_config = {
        "student": {
            "cfg": student_cfg,
            "ckpt": student_ckpt,
            "obs_type": "policy",
        },
        "teacher": {
            "cfg": teacher_cfg,
            "ckpt": teacher_ckpt,
            "obs_type": "expert_policy",
        },
    }

    model_builder.register_network("a2c_aux_depth_enc", A2CWithAuxDepthBuilder)
    model_builder.register_network("a2c_aux_cnn_net", A2CWithAuxCNNBuilder)
    model_builder.register_network("a2c_aux_cnn_net_stereo", A2CWithAuxCNNStereoBuilder)
    model_builder.register_network("a2c_aux_cnn_net_stereo_recon", A2CWithAuxCNNStereoReconBuilder)
    model_builder.register_network("a2c_aux_transformer_stereo", A2CWithAuxTransformerStereoBuilder)
    model_builder.register_network("a2c_aux_transformer_stereo_flow", A2CWithAuxTransformerStereoFlowBuilder)

    dagger = Dagger(env, dagger_config, summaries_dir=summaries_dir, nn_dir=nn_dir)
    dagger.distill()
    if rank == 0:
        dagger.save("dextrah_student_transformer_final")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
