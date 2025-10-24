#!/bin/bash
# 完整的Vision Policy蒸馏训练
# 基于成功的测试运行

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================"
echo "Franka DROID 完整蒸馏训练"
echo "========================================${NC}"

# Teacher checkpoint
TEACHER_RUN="2025-10-23_19-43-40"
TEACHER_CHECKPOINT="model_1499.pt"

# 训练参数（可以根据你的硬件调整）
NUM_ENVS=256
MAX_STEPS=100000
SEED=42
DEVICE="cuda:0"
TASK="Isaac-DROID-Distillation-v0"

echo -e "${BLUE}配置:${NC}"
echo "  Num envs: $NUM_ENVS"
echo "  Max steps: $MAX_STEPS (DEXTRAH-aligned)"
echo "  Beta warmup: 15,000 steps"
echo "  Teacher: $TEACHER_RUN"

# 运行训练
cd /home/huluwulu/Projects/IsaacLab_Manip

source ~/miniconda3/bin/activate env_isaaclab_5.x

python -u scripts/reinforcement_learning/rsl_rl/train_vision_distillation.py \
    --task="$TASK" \
    --num_envs="$NUM_ENVS" \
    --max_steps="$MAX_STEPS" \
    --seed="$SEED" \
    --load_run="$TEACHER_RUN" \
    --checkpoint="$TEACHER_CHECKPOINT" \
    --device="$DEVICE" \
    --headless \
    --enable_cameras 2>&1 | tee full_distillation.log

echo -e "${GREEN}训练完成！${NC}"
echo "Checkpoints保存在: logs/rsl_rl/franka_droid_distillation/"

