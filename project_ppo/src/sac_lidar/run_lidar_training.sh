#!/usr/bin/bash
###############################################################################
# LiDAR SAC Training Launch Script
# This script handles the complete training pipeline:
# 1. Kills existing processes
# 2. Launches Gazebo with small house world
# 3. Starts LiDAR-only SAC training (FOREGROUND)
# 4. Kills Gazebo when training finishes
###############################################################################

set -e  # Exit on error

echo "=============================================="
echo "LiDAR SAC Training Pipeline"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments (optional overrides)
MAX_TIMESTEPS=""
REWARD_TYPE=""
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --max_timesteps)
            MAX_TIMESTEPS="$2"
            shift 2
            ;;
        --reward_type)
            REWARD_TYPE="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--max_timesteps N] [--reward_type TYPE] [--run_name NAME]"
            exit 1
            ;;
    esac
done

# Step 1: Kill all existing processes
echo -e "${YELLOW}[1/4] Killing existing ROS/Gazebo/Python processes...${NC}"
pkill -9 -f 'roslaunch|gzserver|gzclient|rosmaster|python3.*sac' 2>/dev/null || true
sleep 2
echo -e "${GREEN}✓ Processes killed${NC}"

# Step 2: Source ROS environment
echo -e "${YELLOW}[2/4] Setting up ROS environment...${NC}"
source /root/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=/root/catkin_ws/src/aws-robomaker-small-house-world/models
echo -e "${GREEN}✓ ROS environment configured${NC}"

# Step 3: Launch Gazebo with small house world
echo -e "${YELLOW}[3/4] Launching Gazebo with small house world...${NC}"
roslaunch project sac_small_house.launch > /tmp/gazebo_lidar.log 2>&1 &
GAZEBO_PID=$!

# Wait for Gazebo to be ready
echo "Waiting for Gazebo to initialize..."
sleep 12

# Check if Gazebo is running
if ! rosnode list | grep -q "/gazebo"; then
    echo -e "${RED}✗ Gazebo failed to start! Check /tmp/gazebo_lidar.log${NC}"
    kill $GAZEBO_PID 2>/dev/null || true
    exit 1
fi
echo -e "${GREEN}✓ Gazebo running (PID: $GAZEBO_PID)${NC}"

# Step 4: Change to training directory
cd /root/catkin_ws/src/project_ppo/src/sac_lidar

# Create log directory in the models directory (persistent on host)
echo -e "${YELLOW}[4/4] Preparing training...${NC}"
LOG_DIR="models/sac_lidar"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}Training Log:  ${LOG_FILE}${NC}"
echo "=============================================="
echo ""

# Build command with optional overrides
CMD="python3 -u train_sac_lidar.py"
if [ -n "$MAX_TIMESTEPS" ]; then
    CMD="$CMD --max_timesteps $MAX_TIMESTEPS"
fi
if [ -n "$REWARD_TYPE" ]; then
    CMD="$CMD --reward_type $REWARD_TYPE"
fi
if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run_name $RUN_NAME"
fi

echo -e "${GREEN}Starting training (FOREGROUND MODE)...${NC}"
echo "Command: $CMD"
echo "Press Ctrl+C to stop training"
echo "=============================================="
echo ""

# Set up trap to kill Gazebo on exit
cleanup() {
    echo ""
    echo "=============================================="
    echo -e "${YELLOW}Cleaning up...${NC}"
    echo -e "${YELLOW}Killing Gazebo (PID: $GAZEBO_PID)...${NC}"
    kill $GAZEBO_PID 2>/dev/null || true
    pkill -9 -f 'gzserver|gzclient' 2>/dev/null || true
    sleep 2
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    echo "=============================================="
}

trap cleanup EXIT INT TERM

# Run training in FOREGROUND with output to both terminal and log file
$CMD 2>&1 | tee "$LOG_FILE"

# Training finished successfully
TRAIN_EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully${NC}"
else
    echo -e "${RED}✗ Training failed with exit code $TRAIN_EXIT_CODE${NC}"
fi
echo -e "${BLUE}Training log saved: ${LOG_FILE}${NC}"
echo -e "${BLUE}TensorBoard: tensorboard --logdir ${LOG_DIR}/logs${NC}"
echo "=============================================="

# Cleanup happens automatically via trap
exit $TRAIN_EXIT_CODE