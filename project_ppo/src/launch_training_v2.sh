#!/bin/bash
# Production PPO Training Launcher
# Complete end-to-end launcher: ROS + Gazebo + Training with clean run management
# Usage: ./launch_training.sh [--kill_all] [--test_mode] [additional args for main.py]

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Training parameters (override with environment variables)
WORLD="${WORLD:-small_house}"
VISION_BACKBONE="${VISION_BACKBONE:-resnet18}"
USE_MAP_SAMPLER="${USE_MAP_SAMPLER:-true}"
USE_REACHABILITY="${USE_REACHABILITY:-true}"  # Not yet implemented in code, reserved for future
DISTANCE_UNIFORM="${DISTANCE_UNIFORM:-true}"
REWARD_VERSION="${REWARD_VERSION:-V1}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-200000}"
GUI="${GUI:-false}"  # Default to headless for production
KILL_ALL="${KILL_ALL:-false}"
TEST_MODE="${TEST_MODE:-false}"

# Parse command line flags
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --kill_all)
            KILL_ALL="true"
            shift
            ;;
        --test_mode)
            TEST_MODE="true"
            MAX_TIMESTEPS=10000
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Test mode overrides
if [ "$TEST_MODE" = "true" ]; then
    echo "[TEST MODE] Reducing to 10k timesteps for smoke test"
    MAX_TIMESTEPS=10000
fi

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

cleanup_processes() {
    echo "Killing existing processes..."
    pkill -9 -f "main.py" 2>/dev/null || true
    pkill -9 -f "roslaunch.*project_ppo" 2>/dev/null || true
    pkill -9 -f "gzserver" 2>/dev/null || true
    pkill -9 -f "gzclient" 2>/dev/null || true
    pkill -9 -f "rosmaster" 2>/dev/null || true
    pkill -9 -f "roscore" 2>/dev/null || true
    pkill -9 -f "rosout" 2>/dev/null || true
    sleep 2
    echo "✓ Cleanup complete"
}

check_conflicts() {
    local conflicts=0
    if pgrep -f "main.py.*train" >/dev/null 2>&1; then
        echo "⚠ WARNING: Training process already running!"
        conflicts=1
    fi
    if pgrep -f "rosmaster" >/dev/null 2>&1; then
        echo "⚠ WARNING: ROS master already running!"
        conflicts=1
    fi
    if pgrep -f "gzserver" >/dev/null 2>&1; then
        echo "⚠ WARNING: Gazebo server already running!"
        conflicts=1
    fi
    
    if [ $conflicts -eq 1 ]; then
        if [ "$KILL_ALL" = "true" ]; then
            cleanup_processes
        else
            echo ""
            echo "ERROR: Conflicts detected. Use --kill_all to force cleanup or manually kill processes."
            exit 1
        fi
    fi
}

wait_for_ros() {
    echo "Waiting for ROS master..."
    for i in {1..30}; do
        if rostopic list >/dev/null 2>&1; then
            echo "✓ ROS master is ready"
            return 0
        fi
        sleep 1
    done
    echo "✗ ERROR: ROS master did not start"
    return 1
}

wait_for_gazebo() {
    echo "Waiting for Gazebo..."
    for i in {1..60}; do
        if rostopic list 2>/dev/null | grep -q "/gazebo/"; then
            echo "✓ Gazebo is ready"
            return 0
        fi
        sleep 1
    done
    echo "⚠ WARNING: Gazebo topics not detected, but continuing..."
    return 0
}

# ============================================================================
# GENERATE RUN NAME
# ============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M")
VISION_CLEAN=$(echo "$VISION_BACKBONE" | sed 's/_//g')
SAMPLER=$([ "$USE_MAP_SAMPLER" = "true" ] && echo "mapSampler" || echo "randomSampler")
REACH=$([ "$USE_REACHABILITY" = "true" ] && echo "reachBFS" || echo "reachNone")
DIST=$([ "$DISTANCE_UNIFORM" = "true" ] && echo "distUniform" || echo "spatialUniform")
TIMESTEPS_STR=$(python3 -c "
ts = ${MAX_TIMESTEPS}
if ts >= 1000000:
    print(f'T{ts//1000000}M')
elif ts >= 1000:
    print(f'T{ts//1000}k')
else:
    print(f'T{ts}')
")

RUN_NAME="${TIMESTAMP}_${WORLD}_${VISION_CLEAN}_${SAMPLER}_${REACH}_${DIST}_reward${REWARD_VERSION}_${TIMESTEPS_STR}"

# ============================================================================
# SETUP RUN DIRECTORY
# ============================================================================

BASE_RUNS_DIR="/root/catkin_ws/src/project_ppo/src/runs"
RUN_DIR="${BASE_RUNS_DIR}/${RUN_NAME}"

echo "=========================================="
echo "PPO PRODUCTION TRAINING LAUNCHER"
echo "=========================================="
echo "Run Name: ${RUN_NAME}"
echo "World: ${WORLD}"
echo "Vision: ${VISION_BACKBONE}"
echo "Sampler: ${SAMPLER} (map=${USE_MAP_SAMPLER})"
echo "Reachability: ${REACH} (enabled=${USE_REACHABILITY})*"
echo "Distance: ${DIST} (uniform=${DISTANCE_UNIFORM})"
echo "Reward: ${REWARD_VERSION}"
echo "Timesteps: ${MAX_TIMESTEPS} (${TIMESTEPS_STR})"
echo "GUI: ${GUI}"
echo "Test Mode: ${TEST_MODE}"
echo "=========================================="
echo "* Note: Reachability flag is in naming only (not yet implemented in training code)"
echo ""

# Check for conflicts
check_conflicts

# Create run directory structure
mkdir -p "${RUN_DIR}"/{checkpoints,logs,tb,plots}

# Save config
cat > "${RUN_DIR}/config.yml" <<EOF
run_name: ${RUN_NAME}
timestamp: ${TIMESTAMP}
world: ${WORLD}
vision_backbone: ${VISION_BACKBONE}
use_map_sampler: ${USE_MAP_SAMPLER}
use_reachability: ${USE_REACHABILITY}
distance_uniform: ${DISTANCE_UNIFORM}
reward_version: ${REWARD_VERSION}
max_timesteps: ${MAX_TIMESTEPS}
gui: ${GUI}
test_mode: ${TEST_MODE}
EOF

# Create run summary template
cat > "${RUN_DIR}/RUN_SUMMARY.md" <<EOF
# Run Summary: ${RUN_NAME}

**Start Time:** $(date '+%Y-%m-%d %H:%M:%S')

## Configuration

- **World:** ${WORLD}
- **Vision Backbone:** ${VISION_BACKBONE}
- **Sampler Mode:** ${SAMPLER}
- **Reachability:** ${REACH} (naming only, not implemented yet)
- **Distance Sampling:** ${DIST}
- **Reward Version:** ${REWARD_VERSION}
- **Target Timesteps:** ${MAX_TIMESTEPS}
- **GUI:** ${GUI}
- **Test Mode:** ${TEST_MODE}

## Process Information

(To be filled by launcher)

## Results

(To be filled after training completes)

## Notes

(Add observations here)
EOF

echo "✓ Created run directory: ${RUN_DIR}"
echo "  ├── checkpoints/"
echo "  ├── logs/"
echo "  ├── tb/"
echo "  ├── plots/"
echo "  ├── config.yml"
echo "  └── RUN_SUMMARY.md"
echo ""

# ============================================================================
# LAUNCH ROS + GAZEBO
# ============================================================================

echo "=========================================="
echo "LAUNCHING ROS + GAZEBO"
echo "=========================================="

# Source ROS environment
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

# Export TURTLEBOT3_MODEL (required by launch file)
export TURTLEBOT3_MODEL=burger

# Determine GUI argument
GUI_ARG=$([ "$GUI" = "true" ] && echo "gui:=true" || echo "gui:=false")

# Launch ROS + Gazebo in background
echo "Starting roslaunch project_ppo navbot_small_house.launch ${GUI_ARG}..."
nohup roslaunch project_ppo navbot_small_house.launch ${GUI_ARG} \
    > "${RUN_DIR}/logs/ros_gazebo.log" 2>&1 &
ROSLAUNCH_PID=$!
echo "roslaunch PID: ${ROSLAUNCH_PID}"

# Wait for ROS master
if ! wait_for_ros; then
    echo "ERROR: Failed to start ROS master"
    cat "${RUN_DIR}/logs/ros_gazebo.log"
    exit 1
fi

# Wait for Gazebo
wait_for_gazebo

# Get process PIDs
ROSMASTER_PID=$(pgrep -f "rosmaster" || echo "N/A")
GZSERVER_PID=$(pgrep -f "gzserver.*small_house" || echo "N/A")
GZCLIENT_PID=$(pgrep -f "gzclient" || echo "N/A")

echo ""
echo "✓ ROS + Gazebo launched successfully"
echo "  roslaunch PID: ${ROSLAUNCH_PID}"
echo "  rosmaster PID: ${ROSMASTER_PID}"
echo "  gzserver PID: ${GZSERVER_PID}"
echo "  gzclient PID: ${GZCLIENT_PID}"
echo ""

# Save PIDs
cat > "${RUN_DIR}/pids.txt" <<EOF
roslaunch_pid=${ROSLAUNCH_PID}
rosmaster_pid=${ROSMASTER_PID}
gzserver_pid=${GZSERVER_PID}
gzclient_pid=${GZCLIENT_PID}
EOF

# ============================================================================
# LAUNCH TRAINING
# ============================================================================

echo "=========================================="
echo "LAUNCHING TRAINING"
echo "=========================================="

cd /root/catkin_ws/src/project_ppo/src

# Build training command
CMD="python3 main.py"
CMD="$CMD --mode train"
CMD="$CMD --vision_backbone ${VISION_BACKBONE}"
CMD="$CMD --max_timesteps ${MAX_TIMESTEPS}"
CMD="$CMD --timesteps_per_episode 800"
CMD="$CMD --run_name ${RUN_NAME}"

# Add sampler flags
if [ "$USE_MAP_SAMPLER" = "true" ]; then
    CMD="$CMD --use_map_sampler"
fi

if [ "$DISTANCE_UNIFORM" = "true" ]; then
    CMD="$CMD --distance_uniform"
fi

# Add any extra arguments
CMD="$CMD $EXTRA_ARGS"

echo "Command: $CMD"
echo "Logs: ${RUN_DIR}/logs/train.log"
echo ""

# Launch training
nohup $CMD > "${RUN_DIR}/logs/train.log" 2>&1 &
TRAIN_PID=$!

echo "✓ Training started (PID: ${TRAIN_PID})"
echo "train_pid=${TRAIN_PID}" >> "${RUN_DIR}/pids.txt"

# ============================================================================
# UPDATE RUN SUMMARY
# ============================================================================

cat >> "${RUN_DIR}/RUN_SUMMARY.md" <<EOF

## Process Information

- **roslaunch PID:** ${ROSLAUNCH_PID}
- **rosmaster PID:** ${ROSMASTER_PID}
- **gzserver PID:** ${GZSERVER_PID}
- **gzclient PID:** ${GZCLIENT_PID}
- **Training PID:** ${TRAIN_PID}

**Launch Time:** $(date '+%Y-%m-%d %H:%M:%S')

EOF

# ============================================================================
# INITIALIZE/UPDATE GLOBAL COMPARISON TABLE
# ============================================================================

COMPARISON_FILE="${BASE_RUNS_DIR}/RUNS_COMPARISON.md"

if [ ! -f "${COMPARISON_FILE}" ]; then
    echo "Creating global comparison table..."
    cat > "${COMPARISON_FILE}" <<'EOFTABLE'
# Training Runs Comparison

Comprehensive comparison of all PPO training runs for navigation in Gazebo environments.

**Last Updated:** TIMESTAMP_PLACEHOLDER

---

## Runs Table

| Run Name | Date | World | Vision | Sampler | Reach | Dist | Reward | Timesteps | Success % | Collision % | Mean Reward | Checkpoints | Notes |
|----------|------|-------|--------|---------|-------|------|--------|-----------|-----------|-------------|-------------|-------------|-------|

---

## Legend

- **Sampler:** mapSampler (map-based with distance transform) | randomSampler (random in free space)
- **Reach:** reachBFS (reachability checking enabled) | reachNone (no reachability check)  
- **Dist:** distUniform (distance-uniform sampling) | spatialUniform (spatial-uniform sampling)
- **Success %:** Percentage of episodes reaching goal
- **Collision %:** Percentage of episodes ending in collision

---

**Maintained by:** launch_training.sh automation
EOFTABLE
    sed -i "s/TIMESTAMP_PLACEHOLDER/$(date '+%Y-%m-%d %H:%M:%S')/" "${COMPARISON_FILE}"
fi

# Add entry for this run (TBD results)
echo "| ${RUN_NAME} | ${TIMESTAMP} | ${WORLD} | ${VISION_BACKBONE} | ${SAMPLER} | ${REACH} | ${DIST} | ${REWARD_VERSION} | ${MAX_TIMESTEPS} | TBD | TBD | TBD | TBD | **IN PROGRESS** - Started $(date '+%Y-%m-%d %H:%M') |" >> "${COMPARISON_FILE}"

echo "✓ Updated ${COMPARISON_FILE}"
echo ""

# ============================================================================
# MONITORING & FINAL OUTPUT
# ============================================================================

echo "=========================================="
echo "PROCESS SUMMARY"
echo "=========================================="
echo "Run Directory: ${RUN_DIR}"
echo ""
echo "PIDs:"
echo "  roslaunch: ${ROSLAUNCH_PID}"
echo "  rosmaster: ${ROSMASTER_PID}"
echo "  gzserver:  ${GZSERVER_PID}"
echo "  gzclient:  ${GZCLIENT_PID}"
echo "  training:  ${TRAIN_PID}"
echo ""
echo "Logs:"
echo "  Training:    ${RUN_DIR}/logs/train.log"
echo "  ROS/Gazebo:  ${RUN_DIR}/logs/ros_gazebo.log"
echo "  Checkpoints: ${RUN_DIR}/checkpoints/"
echo "  TensorBoard: ${RUN_DIR}/tb/"
echo ""
echo "=========================================="
echo "MONITORING COMMANDS"
echo "=========================================="
echo "Tail training log:"
echo "  tail -f ${RUN_DIR}/logs/train.log"
echo ""
echo "Check processes:"
echo "  pgrep -af 'python.*main.py|rosmaster|gzserver'"
echo ""
echo "Start TensorBoard:"
echo "  tensorboard --logdir=${RUN_DIR}/tb --port=6006 --bind_all"
echo ""
echo "=========================================="
echo "IMPORTANT REMINDERS"
echo "=========================================="
echo "✓ All processes running in background (survive terminal disconnect)"
echo "✓ Closing gzclient is SAFE (rendering only)"
echo "✗ DO NOT kill gzserver (physics simulation required)"
echo "✗ DO NOT kill rosmaster (ROS core required)"
echo "✗ DO NOT kill training process (main.py)"
echo "=========================================="

# Wait for training to initialize and show first output
echo ""
echo "Waiting for training initialization (15 seconds)..."
sleep 15

echo ""
echo "=========================================="
echo "INITIAL TRAINING OUTPUT"
echo "=========================================="
if [ -f "${RUN_DIR}/logs/train.log" ]; then
    tail -n 40 "${RUN_DIR}/logs/train.log"
else
    echo "⚠ Training log not yet created"
fi
echo "=========================================="

echo ""
echo "✅ LAUNCH COMPLETE"
echo ""
echo "Training is running in background."
echo "Monitor with: tail -f ${RUN_DIR}/logs/train.log"
echo ""
echo "Run directory: ${RUN_DIR}"
echo "=========================================="
