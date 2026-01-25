#!/bin/bash
# Test Training Launcher for DINOv2
# Uses baseline config but short runs, saves to runs_test/ directory

set -e

# ============================================================================
# CONFIGURATION - TEST MODE (SHORT RUNS)
# ============================================================================

BASELINE_CONFIG="/root/catkin_ws/src/project_ppo/src/runs/20260124_1606_small_house_dinov2vits14_mapSampler_reachBFS_distUniform_rewardV1_T200k/config.yml"
VISION_BACKBONE="dinov2_vits14"
TIMESTEPS_PER_EPISODE=20  # Short episodes for testing
STEPS_PER_ITERATION=40    # ~2 episodes per iteration
MAX_TIMESTEPS=400         # Total: 10 iterations
SAVE_EVERY_ITERATIONS=2   # Save frequently for testing
USE_MAP_SAMPLER="true"
DISTANCE_UNIFORM="true"
REWARD_TYPE="legacy"
RUNS_ROOT="runs_test"  # Separate directory for test runs

# ============================================================================
# STEP 1: HARD KILL OLD PROCESSES
# ============================================================================

echo "=========================================="
echo "STEP 1: Cleaning up old processes"
echo "=========================================="

pkill -9 -f 'python.*main.py' 2>/dev/null || true
pkill -9 -f 'gzserver' 2>/dev/null || true
pkill -9 -f 'gzclient' 2>/dev/null || true
pkill -9 -f 'roslaunch' 2>/dev/null || true
pkill -9 -f 'rosmaster' 2>/dev/null || true
pkill -9 -f 'roscore' 2>/dev/null || true
pkill -9 -f 'rosout' 2>/dev/null || true

sleep 2

if pgrep -f "gzserver|gzclient|rosmaster|python.*main.py" >/dev/null; then
    echo "ERROR: Failed to kill all processes"
    pgrep -af "gzserver|gzclient|rosmaster|python.*main.py"
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

roslaunch project_ppo navbot_small_house.launch gui:=true > /tmp/gazebo_test.log 2>&1 &
GAZEBO_PID=$!

echo "Gazebo launched (PID: $GAZEBO_PID), waiting for initialization..."

# Wait for Gazebo topics
TIMEOUT=60
ELAPSED=0
until rostopic list 2>/dev/null | grep -q "/scan" && rostopic list 2>/dev/null | grep -q "/clock"; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Gazebo topics not available after ${TIMEOUT}s"
        tail -30 /tmp/gazebo_test.log
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

echo "✓ Gazebo topics available (${ELAPSED}s)"

# ============================================================================
# STEP 4: CAMERA READINESS CHECK (CRITICAL)
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 4: Verifying camera is publishing"
echo "=========================================="

TIMEOUT=30
ELAPSED=0
until rostopic list 2>/dev/null | grep -q "/robot_camera/image_raw"; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Camera topic /robot_camera/image_raw not found after ${TIMEOUT}s"
        rostopic list | grep -i "camera\|image" || echo "No camera topics found!"
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

echo "✓ Camera topic exists"

echo "Checking camera publication rate..."
CAMERA_HZ=$(timeout 10 rostopic hz /robot_camera/image_raw 2>&1 | grep "average rate:" | head -1 | awk '{print $3}')

if [ -z "$CAMERA_HZ" ] || [ "$CAMERA_HZ" = "0" ]; then
    echo "ERROR: Camera not publishing!"
    echo "Gazebo rendering error check:"
    grep -i "CameraSensor\|rendering" /tmp/gazebo_test.log | tail -20
    exit 1
fi

echo "✓ Camera publishing at ~${CAMERA_HZ} Hz"

# ============================================================================
# STEP 5: START TEST TRAINING
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 5: Starting TEST training (short run)"
echo "=========================================="
echo "Using baseline config: $BASELINE_CONFIG"
echo "Test mode: episodes=$TIMESTEPS_PER_EPISODE steps, iterations=40 steps, total=400 steps"
echo "Output directory: runs_test/"
echo ""

cd /root/catkin_ws/src/project_ppo/src

# Build training command
TRAIN_CMD="python3 main.py \
    --mode train \
    --load_config $BASELINE_CONFIG \
    --vision_backbone $VISION_BACKBONE \
    --max_timesteps $MAX_TIMESTEPS \
    --timesteps_per_episode $TIMESTEPS_PER_EPISODE \
    --steps_per_iteration $STEPS_PER_ITERATION \
    --save_every_iterations $SAVE_EVERY_ITERATIONS \
    --runs_root $RUNS_ROOT \
    --reward_type $REWARD_TYPE"

if [ "$USE_MAP_SAMPLER" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --use_map_sampler"
fi

if [ "$DISTANCE_UNIFORM" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --distance_uniform"
fi

echo "Command: $TRAIN_CMD"
echo ""

# Run training
$TRAIN_CMD

echo ""
echo "=========================================="
echo "Test training complete"
echo "=========================================="
echo "Check runs_test/ directory for outputs"
