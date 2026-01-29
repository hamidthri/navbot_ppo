#!/bin/bash
###############################################################################
# LiDAR SAC Training Launch Script
# This script handles the complete training pipeline:
# 1. Kills existing processes
# 2. Launches Gazebo with small house world
# 3. Starts LiDAR-only SAC training
###############################################################################

set -e  # Exit on error

echo "=============================================="
echo "LiDAR SAC Training Pipeline"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Kill all existing processes
echo -e "${YELLOW}[1/4] Killing existing ROS/Gazebo/Python processes...${NC}"
pkill -9 -f 'roslaunch|gzserver|gzclient|rosmaster|python3.*train' 2>/dev/null || true
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
    exit 1
fi
echo -e "${GREEN}✓ Gazebo running (PID: $GAZEBO_PID)${NC}"

# Step 4: Start LiDAR SAC training in background
echo -e "${YELLOW}[4/4] Starting LiDAR SAC training in background...${NC}"
cd /root/catkin_ws/src/project_ppo/src/sac_lidar

# Log file with timestamp
LOG_FILE="/tmp/lidar_training_$(date +%Y%m%d_%H%M%S).log"

echo "Training log: $LOG_FILE"
echo "=============================================="
echo ""

# Run training in background with unbuffered output
nohup python3 -u train_sac_lidar.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo -e "${GREEN}✓ Training started (PID: $TRAIN_PID)${NC}"
echo ""
echo "To monitor training in real-time:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop training:"
echo "  kill $TRAIN_PID"
echo ""
echo "=============================================="
echo "Showing initial training output (5 seconds)..."
sleep 5
tail -30 "$LOG_FILE"
echo "=============================================="
echo -e "${GREEN}Training running in background${NC}"
