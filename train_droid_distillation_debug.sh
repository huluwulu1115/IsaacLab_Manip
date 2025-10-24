#!/bin/bash
# Debug version of distillation training with verbose output

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Franka DROID Vision Distillation (DEBUG)${NC}"
echo -e "${GREEN}========================================${NC}"

# Teacher checkpoint
TEACHER_RUN="2025-10-23_19-43-40"
TEACHER_CHECKPOINT="model_1499.pt"
TEACHER_PATH="logs/rsl_rl/franka_droid_direct/${TEACHER_RUN}/${TEACHER_CHECKPOINT}"

if [ ! -f "$TEACHER_PATH" ]; then
    echo -e "${RED}[ERROR] Teacher checkpoint not found: $TEACHER_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}[INFO] Teacher checkpoint: $TEACHER_PATH${NC}"

# Training parameters (REDUCED for debugging)
NUM_ENVS=16  # Very small to avoid GPU descriptor allocation issues
MAX_STEPS=100  # Just 100 steps for testing (DEXTRAH uses max_steps)
SEED=42
DEVICE="cuda:0"
TASK="Isaac-DROID-Distillation-v0"

echo -e "${GREEN}[START] Debug distillation training${NC}"
echo -e "${BLUE}  Task: $TASK${NC}"
echo -e "${BLUE}  Num envs: $NUM_ENVS (REDUCED for debugging)${NC}"
echo -e "${BLUE}  Max steps: $MAX_STEPS (REDUCED for debugging)${NC}"
echo -e "${BLUE}  Device: $DEVICE${NC}"
echo -e "${BLUE}  Teacher: $TEACHER_RUN${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run with verbose output using isaaclab.sh
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_vision_distillation.py \
    --task="$TASK" \
    --num_envs="$NUM_ENVS" \
    --max_steps="$MAX_STEPS" \
    --seed="$SEED" \
    --device="$DEVICE" \
    --load_run="$TEACHER_RUN" \
    --checkpoint="$TEACHER_CHECKPOINT" \
    --headless \
    --enable_cameras 2>&1 | tee distillation_debug.log

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Debug training completed${NC}"
else
    echo -e "${RED}[FAILED] Training failed with exit code $EXIT_CODE${NC}"
    echo -e "${YELLOW}[INFO] Check distillation_debug.log for details${NC}"
fi

