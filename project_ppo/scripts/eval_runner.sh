#!/bin/bash
################################################################################
# Evaluation Runner - Checkpoint Evaluation with Gazebo GUI
#
# This script evaluates trained PPO checkpoints with Gazebo GUI visible.
# It handles:
# - Gazebo launch with GUI
# - Camera readiness checks
# - Running main.py evaluation for specified checkpoints
# - Saving evaluation CSVs to run logs folder
#
# Usage:
#   ./eval_runner.sh --run_path <PATH> --checkpoint <CKPT_FILE> [OPTIONS]
#
# Examples:
#   ./eval_runner.sh \
#     --run_path "/path/to/runs/method_name" \
#     --checkpoint "actor_iter0013_step00099452.pth" \
#     --eval_episodes 10
#
################################################################################

set -e

# Default parameters
CONTAINER="navbot-ppo"
MAP="small_house"
EVAL_EPISODES=10
CAMERA_TOPIC="/robot_camera/image_raw"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_path)
            RUN_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --eval_episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --map)
            MAP="$2"
            shift 2
            ;;
        --container)
            CONTAINER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --run_path <PATH> --checkpoint <CKPT_FILE> [--eval_episodes N] [--map MAP]"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$RUN_PATH" ]]; then
    echo "ERROR: --run_path is required"
    exit 1
fi

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required"
    exit 1
fi

# Extract method name from run path
METHOD_NAME=$(basename "$RUN_PATH")
RUNS_ROOT=$(dirname "$RUN_PATH")
CHECKPOINT_PATH="$RUN_PATH/checkpoints/$CHECKPOINT"

# Validate checkpoint exists
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    ls -1 "$RUN_PATH/checkpoints/" 2>/dev/null || echo "  (checkpoint directory not found)"
    exit 1
fi

# Load config.yml to get training parameters
CONFIG_FILE="$RUN_PATH/config.yml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config.yml not found: $CONFIG_FILE"
    exit 1
fi

echo "================================================================================"
echo "EVALUATION RUNNER - Checkpoint Evaluation with Gazebo GUI"
echo "================================================================================"
echo "Run Path:         $RUN_PATH"
echo "Method Name:      $METHOD_NAME"
echo "Checkpoint:       $CHECKPOINT"
echo "Eval Episodes:    $EVAL_EPISODES"
echo "Map:              $MAP"
echo "Container:        $CONTAINER"
echo "================================================================================"
echo ""

################################################################################
# Step 1: Hard cleanup - kill any existing ROS/Gazebo/Python processes
################################################################################
echo "[1/6] Hard cleanup: killing all ROS/Gazebo/training processes..."
docker exec $CONTAINER bash -lc "
    pkill -9 -f 'gzserver|gzclient|gazebo|roslaunch|roscore|rosmaster|python3.*main.py' 2>/dev/null || true
    sleep 2
" 2>/dev/null || true

# Also kill from host if running outside container
pkill -9 -f "gzserver|gzclient" 2>/dev/null || true

sleep 2
echo "  ✓ Cleanup complete"
echo ""

################################################################################
# Step 2: Verify DISPLAY and X11 for GUI
################################################################################
echo "[2/6] Verifying DISPLAY and X11 for Gazebo GUI..."
DISPLAY_VAR=$(docker exec $CONTAINER bash -lc 'echo $DISPLAY')
if [[ -z "$DISPLAY_VAR" ]]; then
    echo "  ✗ ERROR: DISPLAY not set in container"
    exit 1
fi
echo "  ✓ DISPLAY set to: $DISPLAY_VAR"

# Check X11 socket
docker exec $CONTAINER bash -lc "ls /tmp/.X11-unix >/dev/null 2>&1" || {
    echo "  ✗ ERROR: /tmp/.X11-unix not accessible in container"
    exit 1
}
echo "  ✓ X11 socket accessible"
echo ""

################################################################################
# Step 3: Launch roscore
################################################################################
echo "[3/6] Starting roscore..."
docker exec -d $CONTAINER bash -lc "
    source /opt/ros/noetic/setup.bash && \
    roscore > /tmp/eval_roscore.log 2>&1
" 2>/dev/null

# Wait for roscore
sleep 3
docker exec $CONTAINER bash -lc "
    source /opt/ros/noetic/setup.bash && \
    timeout 10 bash -c 'until rostopic list &>/dev/null; do sleep 0.5; done'
" || {
    echo "  ✗ ERROR: roscore failed to start"
    exit 1
}
echo "  ✓ roscore ready"
echo ""

################################################################################
# Step 4: Launch Gazebo with GUI
################################################################################
echo "[4/6] Launching Gazebo (${MAP}) with GUI..."
docker exec -d $CONTAINER bash -lc "
    source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    roslaunch project_ppo navbot_small_house.launch gui:=true > /tmp/eval_gazebo.log 2>&1
" 2>/dev/null

# Wait for Gazebo topics
echo "  Waiting for Gazebo topics to be ready..."
docker exec $CONTAINER bash -lc "
    source /opt/ros/noetic/setup.bash && \
    timeout 60 bash -c '
        while true; do
            if rostopic list 2>/dev/null | grep -q \"/scan\" && \
               rostopic list 2>/dev/null | grep -q \"/odom\"; then
                exit 0
            fi
            sleep 1
        done
    '
" || {
    echo "  ✗ ERROR: Gazebo topics not ready after 60s"
    exit 1
}
echo "  ✓ Gazebo ready (/scan and /odom available)"

# Wait for GUI to fully load
sleep 5
echo "  ✓ Gazebo GUI window should be visible now"
echo ""

################################################################################
# Step 5: Verify camera topic
################################################################################
echo "[5/6] Verifying camera topic..."
docker exec $CONTAINER bash -lc "
    source /opt/ros/noetic/setup.bash && \
    rostopic list 2>/dev/null | grep -q '$CAMERA_TOPIC'
" || {
    echo "  ✗ ERROR: Camera topic $CAMERA_TOPIC not found"
    echo ""
    echo "Available topics:"
    docker exec $CONTAINER bash -lc "source /opt/ros/noetic/setup.bash && rostopic list 2>/dev/null | grep camera || true"
    exit 1
}
echo "  ✓ Camera topic $CAMERA_TOPIC exists"

# Check camera Hz
echo "  Checking camera publish frequency..."
CAMERA_HZ=$(docker exec $CONTAINER bash -lc "
    source /opt/ros/noetic/setup.bash && \
    timeout 5 rostopic hz $CAMERA_TOPIC 2>/dev/null | grep 'average rate' | awk '{print \$3}' | head -1
" || echo "0")

if [[ -z "$CAMERA_HZ" ]] || (( $(echo "$CAMERA_HZ < 1" | bc -l) )); then
    echo "  ✗ ERROR: Camera not publishing (Hz: $CAMERA_HZ)"
    exit 1
fi
echo "  ✓ Camera publishing at ~${CAMERA_HZ} Hz"
echo ""

################################################################################
# Step 6: Run evaluation
################################################################################
echo "[6/6] Running evaluation..."
echo "  Checkpoint: $CHECKPOINT"
echo "  Episodes:   $EVAL_EPISODES"
echo ""

# Parse config.yml for training parameters
REWARD_TYPE=$(grep "^reward_type:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}' || echo "fuzzy3")
SAMPLER_MODE=$(grep "^sampler_mode:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}' || echo "rect_regions")
VISION_BACKBONE=$(grep "^vision_backbone:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}' || echo "resnet18")
VISION_PROJ_DIM=$(grep "^vision_proj_dim:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}' || echo "128")
ARCHITECTURE=$(grep "^architecture:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}' || echo "default")
TIMESTEPS_PER_EPISODE=$(grep "^max_timesteps_per_episode:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}' || echo "800")

echo "Training config detected:"
echo "  reward_type:         $REWARD_TYPE"
echo "  sampler_mode:        $SAMPLER_MODE"
echo "  vision_backbone:     $VISION_BACKBONE"
echo "  vision_proj_dim:     $VISION_PROJ_DIM"
echo "  architecture:        $ARCHITECTURE"
echo "  timesteps_per_ep:    $TIMESTEPS_PER_EPISODE"
echo ""

# Create eval log file
EVAL_LOG="$RUN_PATH/logs/eval_${CHECKPOINT%.pth}.log"
mkdir -p "$RUN_PATH/logs"

echo "Starting evaluation (log: $EVAL_LOG)..."
echo "================================================================================"

# Run evaluation in container
docker exec $CONTAINER bash -lc "
    source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    cd /root/catkin_ws/src/project_ppo/src && \
    python3 main.py \
        --mode train \
        --method_name '$METHOD_NAME' \
        --runs_root '$RUNS_ROOT' \
        --reward_type '$REWARD_TYPE' \
        --sampler_mode '$SAMPLER_MODE' \
        --timesteps_per_episode '$TIMESTEPS_PER_EPISODE' \
        --vision_backbone '$VISION_BACKBONE' \
        --vision_proj_dim '$VISION_PROJ_DIM' \
        --architecture '$ARCHITECTURE' \
        --eval \
        --eval_episodes '$EVAL_EPISODES' \
        --actor_model '$CHECKPOINT_PATH' \
        2>&1 | tee '$EVAL_LOG'
"

EVAL_EXIT_CODE=$?

echo "================================================================================"
echo ""

if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    echo "✓ Evaluation completed successfully!"
    echo ""
    echo "Results:"
    echo "  Eval log:    $EVAL_LOG"
    echo "  Eval CSV:    $RUN_PATH/logs/${METHOD_NAME}_eval_episodes.csv"
    echo ""
    
    # Extract metrics from log
    if [[ -f "$EVAL_LOG" ]]; then
        echo "Evaluation summary:"
        grep -A 10 "EVALUATION SUMMARY" "$EVAL_LOG" || echo "  (summary not found in log)"
    fi
else
    echo "✗ ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo ""
    echo "Check log for details: $EVAL_LOG"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Evaluation complete for checkpoint: $CHECKPOINT"
echo "================================================================================"
