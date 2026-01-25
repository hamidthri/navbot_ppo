#!/bin/bash
# Production PPO Training Launcher with Gazebo Camera Verification
# Usage: ./launch_training.sh [--vision_backbone dinov2_vits14] [--max_timesteps 100000]

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

VISION_BACKBONE="${VISION_BACKBONE:-dinov2_vits14}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-100000}"
TIMESTEPS_PER_EPISODE="${TIMESTEPS_PER_EPISODE:-800}"
STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-5000}"
USE_MAP_SAMPLER="true"
DISTANCE_UNIFORM="true"
GUI="true"  # Must be true for camera to work!

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vision_backbone)
            VISION_BACKBONE="$2"
            shift 2
            ;;
        --max_timesteps)
            MAX_TIMESTEPS="$2"
            shift 2
            ;;
        --timesteps_per_episode)
            TIMESTEPS_PER_EPISODE="$2"
            shift 2
            ;;
        --steps_per_iteration)
            STEPS_PER_ITERATION="$2"
            shift 2
            ;;
        --no_gui)
            echo "WARNING: --no_gui disabled camera rendering in past. Using GUI for proper rendering."
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--vision_backbone NAME] [--max_timesteps N]"
            exit 1
            ;;
    esac
done

# ============================================================================
# STEP 1: HARD KILL OLD PROCESSES
# ============================================================================

echo "=========================================="
echo "STEP 1: Cleaning up old processes"
echo "=========================================="

# Kill training processes
pkill -9 -f "python.*main.py" 2>/dev/null || true
pkill -9 -f "python3.*main.py" 2>/dev/null || true

# Kill Gazebo
pkill -9 -f "gzserver" 2>/dev/null || true
pkill -9 -f "gzclient" 2>/dev/null || true

# Kill ROS
pkill -9 -f "roslaunch" 2>/dev/null || true
pkill -9 -f "rosmaster" 2>/dev/null || true
pkill -9 -f "roscore" 2>/dev/null || true
pkill -9 -f "rosout" 2>/dev/null || true

sleep 3

# Verify all killed
if pgrep -f "gzserver|gzclient|rosmaster|main.py" >/dev/null; then
    echo "ERROR: Failed to kill all processes. Please check manually:"
    pgrep -af "gzserver|gzclient|rosmaster|main.py"
    exit 1
fi

echo "✓ All old processes killed"

# ============================================================================
# STEP 2: SOURCE ENVIRONMENT
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 2: Sourcing ROS environment"
echo "=========================================="

source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
export TURTLEBOT3_MODEL=burger
export DISPLAY=${DISPLAY:-:1}

echo "✓ ROS_MASTER_URI=$ROS_MASTER_URI"
echo "✓ TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL"
echo "✓ DISPLAY=$DISPLAY"

# ============================================================================
# STEP 3: LAUNCH GAZEBO WITH GUI (FOR RENDERING)
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 3: Launching Gazebo with rendering"
echo "=========================================="

# Launch in background
roslaunch project_ppo navbot_small_house.launch gui:=$GUI > /tmp/gazebo_launch.log 2>&1 &
GAZEBO_PID=$!

echo "Gazebo launched (PID: $GAZEBO_PID), waiting for initialization..."

# Wait for Gazebo topics
TIMEOUT=60
ELAPSED=0
until rostopic list 2>/dev/null | grep -q "/scan" && rostopic list 2>/dev/null | grep -q "/clock"; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Gazebo topics not available after ${TIMEOUT}s"
        echo "Last 30 lines of Gazebo log:"
        tail -30 /tmp/gazebo_launch.log
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

echo "✓ Gazebo topics available (${ELAPSED}s)"

# Verify gzserver running
if ! pgrep -f "gzserver" >/dev/null; then
    echo "ERROR: gzserver not running!"
    tail -50 /tmp/gazebo_launch.log
    exit 1
fi

# Check gzclient (GUI) if enabled
if [ "$GUI" = "true" ]; then
    if ! pgrep -f "gzclient" >/dev/null; then
        echo "WARNING: gzclient (GUI) not running, but continuing..."
    else
        echo "✓ gzclient (GUI) running"
    fi
fi

# ============================================================================
# STEP 4: CAMERA READINESS CHECK (CRITICAL)
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 4: Verifying camera is publishing"
echo "=========================================="

# Wait for camera topic to exist
TIMEOUT=30
ELAPSED=0
until rostopic list 2>/dev/null | grep -q "/robot_camera/image_raw"; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Camera topic /robot_camera/image_raw not found after ${TIMEOUT}s"
        echo "Available topics:"
        rostopic list | grep -i "camera\|image" || echo "No camera topics found!"
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

echo "✓ Camera topic exists"

# Check if camera is actually publishing (not just topic exists)
echo "Checking camera publication rate..."
CAMERA_HZ=$(timeout 10 rostopic hz /robot_camera/image_raw 2>&1 | grep "average rate:" | head -1 | awk '{print $3}')

if [ -z "$CAMERA_HZ" ] || [ "$CAMERA_HZ" = "0" ]; then
    echo "ERROR: Camera not publishing! (rate=0 or timeout)"
    echo ""
    echo "Gazebo CameraSensor error check:"
    grep -i "CameraSensor\|rendering" /tmp/gazebo_launch.log | tail -20
    echo ""
    echo "This usually means:"
    echo "  1. DISPLAY variable incorrect (current: $DISPLAY)"
    echo "  2. X11 permissions issue (run: xhost +local:docker on host)"
    echo "  3. GUI disabled (gui:=false breaks camera rendering)"
    exit 1
fi

echo "✓ Camera publishing at ~${CAMERA_HZ} Hz"

# ============================================================================
# STEP 5: START TRAINING
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 5: Starting PPO training"
echo "=========================================="

cd /root/catkin_ws/src/project_ppo/src

# Build training command
TRAIN_CMD="python3 main.py \
    --mode train \
    --vision_backbone $VISION_BACKBONE \
    --max_timesteps $MAX_TIMESTEPS \
    --timesteps_per_episode $TIMESTEPS_PER_EPISODE \
    --steps_per_iteration $STEPS_PER_ITERATION"

if [ "$USE_MAP_SAMPLER" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --use_map_sampler"
fi

if [ "$DISTANCE_UNIFORM" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --distance_uniform"
fi

echo "Command: $TRAIN_CMD"
echo ""

# Run training in foreground (so we see output and can Ctrl+C to stop)
$TRAIN_CMD

# ============================================================================
# CLEANUP ON EXIT
# ============================================================================

echo ""
echo "=========================================="
echo "Training complete or interrupted"
echo "=========================================="

# Note: Gazebo will keep running for inspection
# To kill everything: pkill -9 -f "gzserver|gzclient|rosmaster"

echo "Gazebo is still running for inspection."
echo "To kill all: pkill -9 -f 'gzserver|gzclient|rosmaster'"
