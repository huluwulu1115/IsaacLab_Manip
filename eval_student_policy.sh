#!/bin/bash
# Evaluate trained vision student policy

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Vision Student Policy Evaluation${NC}"
echo -e "${GREEN}========================================${NC}"

# Student checkpoint (from training)
STUDENT_RUN="/2025-10-26_13-21-01"  # Your successful training run
STUDENT_CHECKPOINT="model_final.pt"
CHECKPOINT_PATH="logs/rsl_rl/franka_droid_distillation/${STUDENT_RUN}/${STUDENT_CHECKPOINT}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}[WARNING] Checkpoint not found: $CHECKPOINT_PATH${NC}"
    echo -e "${YELLOW}Please update STUDENT_RUN in this script.${NC}"
    echo ""
    echo "Available runs:"
    ls -1 logs/rsl_rl/franka_droid_distillation/ 2>/dev/null || echo "No runs found"
    exit 1
fi

echo -e "${BLUE}[INFO] Checkpoint: $CHECKPOINT_PATH${NC}"

# Evaluation parameters
NUM_ENVS=16       # Small number for better visualization
NUM_EPISODES=50   # Number of episodes to evaluate

echo -e "${BLUE}[INFO] Num envs: $NUM_ENVS${NC}"
echo -e "${BLUE}[INFO] Episodes: $NUM_EPISODES${NC}"

# Run evaluation
cd /home/huluwulu/Projects/IsaacLab_Manip
source ~/miniconda3/bin/activate env_isaaclab_5.x

echo -e "${GREEN}[START] Evaluating student policy...${NC}"

python -u scripts/reinforcement_learning/rsl_rl/play_vision_policy.py \
    --checkpoint="$CHECKPOINT_PATH" \
    --task=Isaac-DROID-Distillation-v0 \
    --num_envs="$NUM_ENVS" \
    --num_episodes="$NUM_EPISODES" \
    --device=cuda:0 \
    --enable_cameras

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Evaluation completed${NC}"
    echo -e "${YELLOW}Results saved in: logs/rsl_rl/franka_droid_distillation/${STUDENT_RUN}/${NC}"
else
    echo -e "${RED}[FAILED] Evaluation failed${NC}"
    exit $EXIT_CODE
fi

