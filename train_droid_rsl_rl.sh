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
readonly TENSORBOARD_PORT=6006
readonly TENSORBOARD_LOG_DIR="logs/rsl_rl/${EXP_NAME}"

# Video generation parameters
readonly VIDEO_LENGTH=500     # frames (~10 seconds)
readonly VIDEO_NUM_ENVS=4     # Number of environments for video
readonly SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
    print_separator
    echo "🎬 Video Generation Options"
    print_separator
    
    # Ask if user wants to generate video
    read -p "Generate video from latest checkpoint? [y/N]: " -n 1 -r GENERATE_VIDEO
    echo ""
    
    if [[ $GENERATE_VIDEO =~ ^[Yy]$ ]]; then
        generate_video_from_latest
    else
        echo "Skipping video generation."
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

# Open video file with available player
open_video() {
    local video_path="$1"
    
    # Skip auto-opening if running over SSH
    if [ "$IS_SSH" = true ]; then
        echo "📹 SSH Mode: Video saved but not opened"
        echo "  Download with: scp $USER@$(hostname):$video_path ."
        echo "  Or use rsync: rsync -avz $USER@$(hostname):$video_path ."
        return 0
    fi
    
    if command -v xdg-open &> /dev/null; then
        xdg-open "$video_path" &> /dev/null &
        echo "✓ Opened with default player (xdg-open)"
        return 0
    elif command -v vlc &> /dev/null; then
        vlc "$video_path" &> /dev/null &
        echo "✓ Opened with VLC"
        return 0
    elif command -v mpv &> /dev/null; then
        mpv "$video_path" &> /dev/null &
        echo "✓ Opened with mpv"
        return 0
    elif command -v ffplay &> /dev/null; then
        ffplay "$video_path" &> /dev/null &
        echo "✓ Opened with ffplay"
        return 0
    else
        echo "⚠ No video player found"
        echo "  Install one: sudo apt install vlc"
        echo "  Or open manually: $video_path"
        return 1
    fi
}

# Print separator line
print_separator() {
    echo "══════════════════════════════════════════════════════════════════════════════"
}

# Generate video from latest checkpoint
generate_video_from_latest() {
    echo ""
    echo "🔍 Looking for latest checkpoint..."
    
    # Find the latest checkpoint
    local latest_ckpt=$(find_best_checkpoint "$EXP_NAME")
    
    if [ -z "$latest_ckpt" ]; then
        echo "⚠ No checkpoint found for experiment: $EXP_NAME"
        echo "   Training may not have saved any checkpoints yet."
        return 1
    fi
    
    local abs_ckpt_path=$(realpath "$latest_ckpt")
    echo "✓ Found checkpoint: $abs_ckpt_path"
    
    # Generate video
    print_separator
    echo "🎬 Generating Policy Visualization Video..."
    print_separator
    printf "Task          : %s\n" "$TASK"
    printf "Checkpoint    : %s\n" "$abs_ckpt_path"
    printf "Video length  : %s frames\n" "$VIDEO_LENGTH"
    printf "Num envs      : %s\n" "$VIDEO_NUM_ENVS"
    print_separator
    
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task="$TASK" \
        --checkpoint="$abs_ckpt_path" \
        --video \
        --enable_cameras \
        --video_length $VIDEO_LENGTH \
        --num_envs=$VIDEO_NUM_ENVS \
        --headless
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Video generated successfully!"
        
        # Find and open the video
        local latest_video=$(find_latest_video)
        if [ -n "$latest_video" ]; then
            local abs_video_path=$(realpath "$latest_video")
            echo "  Video location: $abs_video_path"
            echo ""
            echo "🎬 Opening video..."
            open_video "$abs_video_path"
        fi
    else
        echo ""
        echo "⚠ Video generation failed"
    fi
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

# Find the most recent video in logs directory
find_latest_video() {
    find logs/rsl_rl -name "*.mp4" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2-
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
echo "📋 Next Steps:"
echo "  1. Evaluate the trained policy:"
echo "     ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\"
echo "       --task $TASK --checkpoint '$ABS_CKPT_PATH'"
echo ""
echo "  2. Continue training from this checkpoint:"
echo "     ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\"
echo "       --task $TASK --resume --load_run '$(dirname $CKPT_DIR)'"

#───────────────────────────────────────────────────────────────────────────────────
# POST-TRAINING: VIDEO GENERATION (OPTIONAL)
#───────────────────────────────────────────────────────────────────────────────────

echo ""
print_separator
read -p "📹 Generate video of trained policy? [y/N]: " -n 1 -r GENERATE_VIDEO
echo ""

if [[ ! $GENERATE_VIDEO =~ ^[Yy]$ ]]; then
    echo "Skipping video generation."
    echo ""
    echo "📊 TensorBoard is still running at: http://localhost:$TENSORBOARD_PORT"
    echo "   To stop it: pkill -f tensorboard"
    print_separator
    exit 0
fi

# Generate video
print_separator
echo "🎬 Generating Policy Visualization Video..."
print_separator
printf "Task          : %s\n" "$TASK"
printf "Checkpoint    : %s\n" "$ABS_CKPT_PATH"
printf "Video length  : %s frames\n" "$VIDEO_LENGTH"
printf "Num envs      : %s\n" "$VIDEO_NUM_ENVS"
print_separator

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task="$TASK" \
    --checkpoint="$ABS_CKPT_PATH" \
    --video \
    --enable_cameras \
    --video_length $VIDEO_LENGTH \
    --num_envs=$VIDEO_NUM_ENVS \
    --headless

# Check if video generation succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠ Video generation failed"
    print_separator
    exit 1
fi

# Find and open the generated video
echo ""
echo "✓ Video generated successfully!"

LATEST_VIDEO=$(find_latest_video)

if [ -z "$LATEST_VIDEO" ]; then
    echo "⚠ Could not locate generated video in logs/"
    print_separator
    exit 1
fi

ABS_VIDEO_PATH=$(realpath "$LATEST_VIDEO")
echo "  Video location: $ABS_VIDEO_PATH"

# Open video
echo ""
echo "🎬 Opening video..."
open_video "$ABS_VIDEO_PATH"

print_separator
echo "✨ All done! Enjoy your trained policy!"
echo ""
echo "📊 TensorBoard is still running at: http://localhost:$TENSORBOARD_PORT"
echo "   To stop it: pkill -f tensorboard"
print_separator

