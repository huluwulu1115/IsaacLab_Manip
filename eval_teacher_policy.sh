#!/bin/bash
# Evaluate teacher policy as baseline with optional early termination

echo "========================================="
echo "Teacher Policy Evaluation (Baseline)"
echo "========================================="

cd /home/huluwulu/Projects/IsaacLab_Manip

# Configuration
TEACHER_RUN="2025-11-13_22-43-54"
TEACHER_CHECKPOINT="model_1499.pt"
CHECKPOINT_PATH="logs/rsl_rl/franka_droid_cube_direct/${TEACHER_RUN}/${TEACHER_CHECKPOINT}"

# Early termination settings (default: enabled)
ENABLE_EARLY_TERMINATION=${ENABLE_EARLY_TERMINATION:-true}

echo "Teacher: $CHECKPOINT_PATH"
echo "Early Termination: $ENABLE_EARLY_TERMINATION"
echo "Running evaluation..."

# Run RSL-RL play script for teacher (use full absolute path)
FULL_CHECKPOINT_PATH="$(pwd)/$CHECKPOINT_PATH"

# Build command with optional early termination flag
if [ "$ENABLE_EARLY_TERMINATION" = "true" ]; then
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task=Isaac-DROID-Direct-v0 \
        --num_envs=16 \
        --num_episodes=50 \
        --checkpoint="$FULL_CHECKPOINT_PATH" \
        --enable_early_termination
else
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task=Isaac-DROID-Direct-v0 \
        --num_envs=16 \
        --num_episodes=50 \
        --checkpoint="$FULL_CHECKPOINT_PATH"
fi

echo "========================================="
echo "Check the console output above for teacher performance"
echo "Early termination allows episodes to end once success is achieved"
echo "========================================="

