#!/bin/bash
# RSL-RL Vision Distillation Training

set -e

########################################################################################
# Teacher
TEACHER_RUN="2025-10-23_19-43-40"
TEACHER_CHECKPOINT="model_1499.pt"

# Training params
NUM_ENVS=128
MAX_STEPS=100000
SEED=42
DEVICE="cuda:0"
TASK="Isaac-DROID-Distillation-v0"

# Early Termination (true/false)
EARLY_TERMINATION=true

# Beta Schedule (Quickly modify to test different strategies)
# Current Strategy: Fast transition + long imitation (15k-35k transition, 35k-100k pure student)
BETA_START=15000
BETA_END=35000    # 20k steps fast transition
# Other strategies examples:
# Conservative: BETA_START=15000, BETA_END=60000 (45k steps transition)
# Super fast: BETA_START=10000, BETA_END=25000 (15k steps transition)
# Faster: BETA_START=15000, BETA_END=30000 (15k steps transition)


# Target std for student (overrides teacher's high exploration std=2.4)
TARGET_STD=0.3
# 0.2-0.3: Precise control (recommended for grasping)
# 0.4-0.5: Moderate exploration
# 0.6-0.8: Higher exploration

# Loss type (true/false)
USE_KL_LOSS=false
# true:  KL Divergence (information-theoretic, theoretically optimal)
# false: DEXTRAH Weighted MSE (default, simpler and stable)

# Video recording (true/false) - WARNING: Slows training ~30%
ENABLE_VIDEO=true
VIDEO_INTERVAL=10000  # Record every 10k steps
VIDEO_LENGTH=200      # Steps per video (enough for 2 complete episodes ~250 steps each)
# true:  Record viewport videos during training (third-person view)
# false: No videos, use play_vision_policy.py after training
########################################################################################

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Cleaning GPU and memory${NC}"
echo -e "${YELLOW}========================================${NC}"

pkill -9 -f "python.*Isaac" 2>/dev/null || true
sleep 2

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo -e "${GREEN}✅ GPU cleaned${NC}\n"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Franka DROID Vision Distillation${NC}"
echo -e "${GREEN}========================================${NC}"


echo -e "${BLUE}Parameters:${NC}"
echo "  Task: $TASK"
echo "  Num envs: $NUM_ENVS"
echo "  Max steps: $MAX_STEPS"
echo "  Teacher: $TEACHER_RUN"
echo "  Beta schedule: ${BETA_START}-${BETA_END} ($((BETA_END - BETA_START)) steps transition)"
echo "  Target std: $TARGET_STD (overriding teacher's exploration std)"
echo "  Loss type: $(if [ "$USE_KL_LOSS" = "true" ]; then echo "KL Divergence"; else echo "DEXTRAH Weighted MSE"; fi)"
echo "  Early termination: $(if [ "$EARLY_TERMINATION" = "true" ]; then echo "Enabled"; else echo "Disabled"; fi)"
echo "  Video recording: $(if [ "$ENABLE_VIDEO" = "true" ]; then echo "Enabled (every ${VIDEO_INTERVAL} steps, ${VIDEO_LENGTH} steps/video)"; else echo "Disabled"; fi)"
echo "  Logging: W&B (optional)"
echo ""

cd /home/huluwulu/Projects/IsaacLab_Manip
source ~/miniconda3/bin/activate env_isaaclab_5.x

echo -e "${GREEN}[START] Starting training...${NC}\n"

# Build command with optional flags
CMD="python -u scripts/reinforcement_learning/rsl_rl/train_vision_distillation.py \
    --task=$TASK \
    --num_envs=$NUM_ENVS \
    --max_steps=$MAX_STEPS \
    --seed=$SEED \
    --load_run=$TEACHER_RUN \
    --checkpoint=$TEACHER_CHECKPOINT \
    --beta_start_decay=$BETA_START \
    --beta_end_decay=$BETA_END \
    --target_std=$TARGET_STD"

# Add early termination flag if enabled
if [ "$EARLY_TERMINATION" = "true" ]; then
    CMD="$CMD --early_termination"
fi

# Add KL loss flag if enabled
if [ "$USE_KL_LOSS" = "true" ]; then
    CMD="$CMD --use_kl_loss"
fi

# Add video recording flags if enabled
if [ "$ENABLE_VIDEO" = "true" ]; then
    CMD="$CMD --video --video_interval=$VIDEO_INTERVAL --video_length=$VIDEO_LENGTH"
fi

# Add rendering flags
CMD="$CMD --headless --enable_cameras"

# Run with logging
eval $CMD 2>&1 | tee distillation_training.log

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Training completed${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "Checkpoints: logs/rsl_rl/franka_droid_distillation/"
else
    echo -e "${YELLOW}✗ Training failed (Exit code: $EXIT_CODE)${NC}"
fi

echo ""
pkill -9 -f "python.*train" 2>/dev/null || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo -e "${GREEN}✅ Cleaning completed${NC}"

