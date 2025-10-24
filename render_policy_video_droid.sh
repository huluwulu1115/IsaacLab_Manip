echo "Starting Visualization"

# visualization parameters
TASK="Isaac-DROID-Direct-v1"
GPU_ID=0                                

export task=$TASK
export checkpoint_path="/home/huluwulu/Projects/IsaacLab_Manip/logs/rl_games/franka_droid_direct/2025-10-22_22-40-39/nn/franka_droid_direct.pth"
export num_envs=1
export env_video_size="480 640"
export video_length=500

# Camera parameters (note: camera_position is not currently supported by play.py)
export camera_position="0.6 0.9 1.0"

echo "task: $TASK"
echo "checkpoint_path: $checkpoint_path"
echo "video_length: $video_length"
echo "num_envs: $num_envs"
echo "env_video_size: $env_video_size"
echo "camera_position: $camera_position"

# paths
export isaaclab_eureka_root_dir="/home/huluwulu/Projects/IsaacLab_Manip/IsaacLabEureka/logs"
export TMPDIR="/home/huluwulu/Projects/IsaacLab_Manip/tmp"
echo "isaaclab_eureka_root_dir: $isaaclab_eureka_root_dir"
echo "TMPDIR: $TMPDIR"


python /home/huluwulu/Projects/IsaacLab_Manip/scripts/reinforcement_learning/rl_games/play.py \
--task=$TASK \
--checkpoint=$checkpoint_path \
--video \
--enable_cameras \
--video_length $video_length \
--num_envs=$num_envs \

echo "done"

