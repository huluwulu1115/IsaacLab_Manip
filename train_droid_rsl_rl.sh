#!/usr/bin/env bash

set -euo pipefail  # Exit on error, undefined variable, or pipe failure

#───────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
#───────────────────────────────────────────────────────────────────────────────────

# Training parameters
readonly TASK="Isaac-DROID-Direct-v0"
readonly NUM_ENVS=4096
readonly EXP_NAME="franka_droid_direct"
readonly MAX_ITERATIONS=1500
readonly SEED=42

# Environment setup
readonly GPU_ID=0
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TMPDIR="$(pwd)/tmp"

# TensorBoard parameters
readonly TENSORBOARD_PORT=6007
readonly TENSORBOARD_LOG_DIR="logs/rsl_rl/${EXP_NAME}"

# Detect if running over SSH
IS_SSH=false
if [ -n "${SSH_CLIENT:-}" ] || [ -n "${SSH_TTY:-}" ] || [ -n "${SSH_CONNECTION:-}" ]; then
    IS_SSH=true
fi

#───────────────────────────────────────────────────────────────────────────────────
# SIGNAL HANDLING (Ctrl+C)
#───────────────────────────────────────────────────────────────────────────────────

# Global flag to track if we're in training
TRAINING_ACTIVE=false
TRAINING_PID=""

# Handle Ctrl+C gracefully
handle_interrupt() {
    echo ""
    echo "⚠️  Training interrupted by user (Ctrl+C)"
    
    # Kill the training process if it's still running
    if [ -n "$TRAINING_PID" ] && ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        echo "🛑 Stopping training process..."
        kill "$TRAINING_PID" 2>/dev/null || true
        wait "$TRAINING_PID" 2>/dev/null || true
    fi
    
    echo ""
    echo "📊 TensorBoard is still running at: http://localhost:$TENSORBOARD_PORT"
    echo "   To stop it: pkill -f tensorboard"
    print_separator
    exit 130  # Standard exit code for Ctrl+C
}

# Set up trap for SIGINT (Ctrl+C)
trap handle_interrupt SIGINT

#───────────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
#───────────────────────────────────────────────────────────────────────────────────

# Print separator line
print_separator() {
    echo "══════════════════════════════════════════════════════════════════════════════"
}

# Find the best checkpoint from RSL-RL training
find_best_checkpoint() {
    local exp_name="$1"
    # RSL-RL stores checkpoints in logs/rsl_rl/<exp_name>/<timestamp>/model_*.pt
    local exp_dir=$(find "logs/rsl_rl/${exp_name}" -type d -maxdepth 1 2>/dev/null | grep -E '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}' | sort -r | head -n1)
    
    if [ -n "$exp_dir" ]; then
        # Find the latest model checkpoint (usually the best for on-policy algorithms)
        find "$exp_dir" -type f -name "model_*.pt" 2>/dev/null | sort -V | tail -n1
    fi
}

# Launch TensorBoard in the background
launch_tensorboard() {
    local log_dir="$1"
    local port="$2"
    
    # Check if TensorBoard is already running on this port
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠ Port $port already in use (TensorBoard may already be running)"
        if [ "$IS_SSH" = true ]; then
            echo "  📊 Access at: http://$(hostname):$port"
        else
            echo "  📊 Access at: http://localhost:$port"
        fi
        return 0
    fi
    
    # Check if tensorboard command is available
    if ! command -v tensorboard &> /dev/null; then
        echo "⚠ TensorBoard not found (install: pip install tensorboard)"
        return 1
    fi
    
    # Create log directory if it doesn't exist
    mkdir -p "$log_dir"
    
    # Launch TensorBoard in the background
    tensorboard --logdir="$log_dir" --port=$port --host=0.0.0.0 &> /dev/null &
    local tb_pid=$!
    
    # Wait a moment for TensorBoard to start
    sleep 2
    
    # Check if it's actually running
    if ps -p $tb_pid > /dev/null 2>&1; then
        echo "✓ TensorBoard started (PID: $tb_pid)"
        if [ "$IS_SSH" = true ]; then
            echo "  📊 Access at: http://$(hostname):$port"
            echo "  Or use: http://localhost:$port (if on VPN/same network)"
        else
            echo "  📊 Access at: http://localhost:$port"
        fi
        return 0
    else
        echo "⚠ TensorBoard failed to start"
        return 1
    fi
}

#───────────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING
#───────────────────────────────────────────────────────────────────────────────────

print_separator
echo "🚀 RL Training - RSL-RL"
print_separator
printf "Framework     : RSL-RL\n"
printf "GPU           : %s\n" "$GPU_ID"
printf "Task          : %s\n" "$TASK"
printf "Experiment    : %s\n" "$EXP_NAME"
printf "Num envs      : %s\n" "$NUM_ENVS"
printf "Max iterations: %s\n" "$MAX_ITERATIONS"
printf "Seed          : %s\n" "$SEED"
if [ "$IS_SSH" = true ]; then
    printf "Connection    : SSH (remote mode)\n"
fi
print_separator

# Launch TensorBoard
echo ""
echo "📊 Starting TensorBoard..."
echo "   Watching all '$EXP_NAME' experiments (current + history)"
echo "   Directory: $TENSORBOARD_LOG_DIR"
launch_tensorboard "$TENSORBOARD_LOG_DIR" "$TENSORBOARD_PORT"
echo ""

# Run training with RSL-RL
echo "🚀 Training with RSL-RL..."
echo "   Config: source/isaaclab_tasks/isaaclab_tasks/direct/franka_droid/agents/rsl_rl_ppo_cfg.py"
echo "   Press Ctrl+C to stop training early"
echo ""

# Set training active flag
TRAINING_ACTIVE=true

# Run training and capture PID
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --seed "$SEED" \
    --headless &

TRAINING_PID=$!

# Wait for training to complete
wait $TRAINING_PID 2>/dev/null
TRAIN_EXIT_CODE=$?

# Reset flag
TRAINING_ACTIVE=false

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
else
    echo ""
    echo "⚠ Training exited with code: $TRAIN_EXIT_CODE"
fi

#───────────────────────────────────────────────────────────────────────────────────
# POST-TRAINING: CHECKPOINT MANAGEMENT
#───────────────────────────────────────────────────────────────────────────────────

echo ""
print_separator
echo "📦 Finding Best Checkpoint..."
print_separator

BEST_CKPT=$(find_best_checkpoint "$EXP_NAME")

if [ -z "$BEST_CKPT" ]; then
    echo "⚠ No checkpoint found for experiment: $EXP_NAME"
    echo "   Searching in: logs/rsl_rl/${EXP_NAME}/"
    echo ""
    echo "   Available runs:"
    find "logs/rsl_rl/${EXP_NAME}" -type d -maxdepth 1 2>/dev/null | grep -E '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -5
    exit 1
fi

ABS_CKPT_PATH=$(realpath "$BEST_CKPT")
CKPT_DIR=$(dirname "$ABS_CKPT_PATH")
echo "✓ Latest checkpoint found:"
echo "  $ABS_CKPT_PATH"
echo ""
echo "  Checkpoint directory:"
echo "  $CKPT_DIR"

# Show next steps
echo ""
print_separator
echo "📋 How to Use Your Trained Policy:"
print_separator
echo ""
echo "1️⃣  Evaluate policy (GUI mode):"
echo ""
echo "    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\"
echo "      --task $TASK \\"
echo "      --checkpoint '$ABS_CKPT_PATH' \\"
echo "      --num_envs 16"
echo ""
echo "2️⃣  Evaluate policy with video recording:"
echo ""
echo "    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\"
echo "      --task $TASK \\"
echo "      --checkpoint '$ABS_CKPT_PATH' \\"
echo "      --num_envs 4 \\"
echo "      --video \\"
echo "      --video_length 500 \\"
echo "      --headless"
echo ""
echo "3️⃣  Continue training from this checkpoint:"
echo ""
echo "    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\"
echo "      --task $TASK \\"
echo "      --resume \\"
echo "      --load_run '$(dirname $CKPT_DIR)'"
echo ""
print_separator
echo "📊 TensorBoard is still running at: http://localhost:$TENSORBOARD_PORT"
echo "   To stop it: pkill -f tensorboard"
print_separator

