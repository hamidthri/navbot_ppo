#!/bin/bash
set -e

echo "=========================================="
echo "Small House GUI Verification Script"
echo "=========================================="

# 1) Kill everything hard
echo "[1/6] Killing all processes..."
pkill -9 -f "gzserver|gzclient|gazebo|roscore|rosmaster|rosout|roslaunch|python3.*main.py" 2>/dev/null || true
sleep 2

# 2) Source environment
echo "[2/6] Sourcing ROS environment..."
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
export TURTLEBOT3_MODEL=burger
export DISPLAY=:1

# 3) Start roscore explicitly
echo "[3/6] Starting roscore..."
roscore >/tmp/roscore.log 2>&1 &
ROSCORE_PID=$!
echo "roscore PID: $ROSCORE_PID"

# Wait for roscore to be ready
until rostopic list >/dev/null 2>&1; do 
    sleep 0.2
done
echo "roscore ready!"

# 4) Launch Small House with GUI
echo "[4/6] Launching Small House world with GUI..."
roslaunch project_ppo navbot_small_house.launch gui:=true >/tmp/gazebo_small_house.log 2>&1 &
GAZEBO_PID=$!
echo "Gazebo launch PID: $GAZEBO_PID"

# Wait for Gazebo topics (with timeout)
echo "Waiting for Gazebo topics..."
TIMEOUT=60
ELAPSED=0
until rostopic list | grep -q "/scan" && rostopic list | grep -q "/clock"; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Gazebo topics not available after ${TIMEOUT}s"
        exit 1
    fi
done
echo "Gazebo topics ready! (waited ${ELAPSED}s)"

# Give Gazebo GUI time to fully initialize
sleep 3

# 5) Run SHORT training verification
echo "[5/6] Running training verification (800 timesteps)..."
cd /root/catkin_ws/src/project_ppo/src

python3 main.py \
    --mode train \
    --method_name verify_small_house_gui \
    --vision_backbone resnet18 \
    --vision_proj_dim 64 \
    --timesteps_per_episode 50 \
    --steps_per_iteration 400 \
    --max_timesteps 800 \
    --use_map_sampler \
    --distance_uniform \
    > /tmp/verify_small_house_gui.log 2>&1

echo "Training verification completed!"

# 6) Print summary
echo ""
echo "=========================================="
echo "VERIFICATION RESULTS"
echo "=========================================="

echo ""
echo "Package verification:"
rospack find project_ppo

echo ""
echo "Process status:"
echo "  roscore PID: $ROSCORE_PID ($(ps -p $ROSCORE_PID -o comm= 2>/dev/null || echo 'NOT RUNNING'))"
GZSERVER_PID=$(pgrep gzserver || echo "NONE")
echo "  gzserver PID: $GZSERVER_PID"
GZCLIENT_PID=$(pgrep gzclient || echo "NONE")
echo "  gzclient PID: $GZCLIENT_PID"

echo ""
echo "Training log (last 30 lines):"
echo "----------------------------------------"
tail -n 30 /tmp/verify_small_house_gui.log

echo ""
echo "=========================================="
echo "Verification complete!"
echo "=========================================="
