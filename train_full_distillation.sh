#!/bin/bash
# 完整的Vision Policy蒸馏训练
# 基于成功的测试运行

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}========================================"
echo "清理 GPU 和内存"
echo "========================================${NC}"

# 终止所有 Isaac/Python 进程
pkill -9 -f "python.*Isaac" 2>/dev/null || true
pkill -9 -f "train_vision" 2>/dev/null || true
pkill -9 -f "kit.*Isaac" 2>/dev/null || true

sleep 2

# 检查 GPU 内存
echo -e "${BLUE}GPU 状态:${NC}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader,nounits | while read line; do
    echo "  $line MB"
done

echo ""

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

echo -e "${GREEN}========================================"
echo "训练完成！"
echo "========================================${NC}"
echo "Checkpoints保存在: logs/rsl_rl/franka_droid_distillation/"
echo ""
echo -e "${YELLOW}清理训练进程...${NC}"
pkill -9 -f "python.*train" 2>/dev/null || true
sleep 1
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo -e "${GREEN}✅ 清理完成${NC}"

