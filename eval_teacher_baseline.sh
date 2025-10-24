#!/bin/bash
# Evaluate teacher policy as baseline

echo "========================================="
echo "Teacher Policy Evaluation (Baseline)"
echo "========================================="

cd /home/huluwulu/Projects/IsaacLab_Manip

# Teacher checkpoint
TEACHER_RUN="2025-10-23_19-43-40"
TEACHER_CHECKPOINT="model_1499.pt"
CHECKPOINT_PATH="logs/rsl_rl/franka_droid_direct/${TEACHER_RUN}/${TEACHER_CHECKPOINT}"

echo "Teacher: $CHECKPOINT_PATH"
echo "Running evaluation..."

# Run RSL-RL play script for teacher (use full absolute path)
FULL_CHECKPOINT_PATH="$(pwd)/$CHECKPOINT_PATH"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-DROID-Direct-v0 \
    --num_envs=16 \
    --num_episodes=50 \
    --checkpoint="$FULL_CHECKPOINT_PATH"

echo "========================================="
echo "Check the console output above for teacher performance"
echo "========================================="

