#!/usr/bin/env bash

set -euo pipefail  # Exit on error, undefined variable, or pipe failure

#───────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
#───────────────────────────────────────────────────────────────────────────────────

# Training parameters
readonly TASK="Isaac-DROID-Direct-v0"
readonly NUM_ENVS=4096
readonly EXP_NAME="franka_droid_direct"
readonly MAX_ITERATIONS=1500  # Overrides config file value (config has 3000, but command line takes precedence)
readonly SEED=42

# Video recording parameters (set ENABLE_VIDEO=true to record videos)
readonly ENABLE_VIDEO=false  # Set to true to enable video recording
readonly VIDEO_INTERVAL=500   # Record video every 500 steps
readonly VIDEO_LENGTH=200     # Length of each video in steps

# Environment setup
readonly GPU_ID=0
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TMPDIR="$(pwd)/tmp"

# Wandb parameters
readonly WANDB_PROJECT="franka_droid_teacher"

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
    echo "📊 Training logs are available on Wandb"
    echo "   Project: $WANDB_PROJECT"
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

# Check if wandb is configured
check_wandb() {
    if command -v wandb >/dev/null 2>&1; then
        # Check if wandb is logged in
        if wandb status >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}


#───────────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING
#───────────────────────────────────────────────────────────────────────────────────

print_separator
echo "🎓 Teacher Policy Training - RSL-RL PPO"
print_separator
printf "Policy Type   : Teacher (State-based, Privileged Info)\n"
printf "Framework     : RSL-RL\n"
printf "Algorithm     : PPO (Proximal Policy Optimization)\n"
printf "GPU           : %s\n" "$GPU_ID"
printf "Task          : %s\n" "$TASK"
printf "Experiment    : %s\n" "$EXP_NAME"
printf "Num envs      : %s\n" "$NUM_ENVS"
printf "Max iterations: %s\n" "$MAX_ITERATIONS"
printf "Seed          : %s\n" "$SEED"
printf "Logger        : Wandb\n"
printf "Wandb Project : %s\n" "$WANDB_PROJECT"
if [ "$ENABLE_VIDEO" = true ]; then
    printf "Video Recording: Enabled (every %d steps, %d steps/video)\n" "$VIDEO_INTERVAL" "$VIDEO_LENGTH"
    printf "Video Upload  : Auto-upload to Wandb after training\n"
else
    printf "Video Recording: Disabled\n"
fi
if [ "$IS_SSH" = true ]; then
    printf "Connection    : SSH (remote mode)\n"
fi
print_separator

# Check Wandb configuration
echo ""
echo "📊 Checking Wandb configuration..."
check_wandb || {
    echo ""
    echo "⚠️  Wandb is not configured. Please run:"
    echo "    pip install wandb"
    echo "    wandb login"
    echo ""
    exit 1
}
echo ""

# Run training with RSL-RL
echo "🚀 Training with RSL-RL..."
echo "   Config: source/isaaclab_tasks/isaaclab_tasks/direct/franka_droid/agents/rsl_rl_ppo_cfg.py"
echo "   Press Ctrl+C to stop training early"
echo ""

# Set training active flag
TRAINING_ACTIVE=true

# Build training command
TRAIN_CMD=(
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py
    --task "$TASK"
    --num_envs "$NUM_ENVS"
    --max_iterations "$MAX_ITERATIONS"
    --seed "$SEED"
    --headless
)

# Add video recording if enabled
if [ "$ENABLE_VIDEO" = true ]; then
    TRAIN_CMD+=(--video)
    TRAIN_CMD+=(--video_interval "$VIDEO_INTERVAL")
    TRAIN_CMD+=(--video_length "$VIDEO_LENGTH")
fi

# Run training and capture PID
"${TRAIN_CMD[@]}" &

TRAINING_PID=$!

# Wait for training to complete
wait $TRAINING_PID 2>/dev/null
TRAIN_EXIT_CODE=$?

# Reset flag
TRAINING_ACTIVE=false

# Check training result
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
else
    echo ""
    echo "⚠️  Training exited with code: $TRAIN_EXIT_CODE"
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

# Show video location if enabled
if [ "$ENABLE_VIDEO" = true ]; then
    print_separator
    echo "📹 Training Videos:"
    print_separator
    VIDEO_DIR="$CKPT_DIR/videos/train"
    if [ -d "$VIDEO_DIR" ]; then
        VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l)
        echo "  Video directory: $VIDEO_DIR"
        echo "  Videos recorded: $VIDEO_COUNT"
        echo ""
        echo "  ℹ️  Videos are saved locally every $VIDEO_INTERVAL steps"
        echo "  ℹ️  Video length: $VIDEO_LENGTH steps per video"
        echo "  ℹ️  Videos are automatically uploaded to Wandb after training"
    else
        echo "  ⚠️  Video directory not found: $VIDEO_DIR"
        echo "  Videos may not have been recorded during training."
    fi
    echo ""
fi

# Show next steps
echo ""
print_separator
echo "📋 How to Use Your Trained Teacher Policy:"
print_separator
echo ""
echo "1️⃣  Evaluate policy (GUI mode):"
echo ""
echo "    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\"
echo "      --task $TASK \\"
echo "      --checkpoint \"$ABS_CKPT_PATH\" \\"
echo "      --num_envs 16"
echo ""
echo "2️⃣  Use this teacher for distillation training:"
echo ""
echo "    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_vision_distillation.py \\"
echo "      --task Isaac-DROID-Distillation-v0 \\"
echo "      --load_run \"$(basename "$CKPT_DIR")\" \\"
echo "      --checkpoint \"$(basename "$ABS_CKPT_PATH")\" \\"
echo "      --num_envs 128 \\"
echo "      --max_steps 100000 \\"
echo "      --early_termination"
echo ""
print_separator
echo "📊 Training logs available on Wandb:"
echo "   Project: $WANDB_PROJECT"
echo "   URL: https://wandb.ai/<your-username>/${WANDB_PROJECT}"
print_separator
