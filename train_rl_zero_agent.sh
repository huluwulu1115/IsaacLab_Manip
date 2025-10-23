#!/usr/bin/env bash
set -euo pipefail                       # abort on any error, unset var, or pipe fail

# ─── User‑configurable parameters ────────────────────────────────────────────────
GPU_ID=0                                # pick the physical GPU

# TASK="Isaac-Droid-Stapler-Grasplift-Direct-v0"
TASK="Isaac-DROID-Direct-v0"
NUM_ENVS=1
# ────────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${GPU_ID}" # expose only the chosen GPU to Isaac Sim
export TMPDIR="/home/huluwulu/Projects/IsaacLab_Manip/tmp"

printf "GPU           : %s\nTask          : %s\nNum envs      : %s\n" \
       "$GPU_ID" "$TASK" "$NUM_ENVS"
echo "Starting RL experiment …"
echo "──────────────────────────────────────────────────────────────────────────────"

python scripts/environments/zero_agent.py \
  --task "$TASK" \
  --num_envs "$NUM_ENVS" \
  --enable_cameras


echo "Done ✔︎"
