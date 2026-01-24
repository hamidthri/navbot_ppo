#!/bin/bash
# Production Training Script: 200k timesteps with Steps=1 fix
# Small House + ResNet18 + Distance-Uniform Sampling

set -e

echo "=========================================="
echo "Production Training: 200k Timesteps"
echo "Small House + ResNet18 + Steps=1 Fix"
echo "=========================================="
echo ""

CONTAINER="navbot-ppo"
PROJECT_DIR="/root/catkin_ws/src/project_ppo"
RUN_NAME="small_house_resnet18_200k_steps1fix_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/tmp/train_${RUN_NAME}.log"

echo "Run name: $RUN_NAME"
echo "Log file: $LOG_FILE"
echo ""

# Check if running inside container
if [ ! -f /opt/ros/noetic/setup.bash ]; then
    echo "ERROR: This script must run inside the navbot-ppo container"
    echo "Usage: docker exec navbot-ppo bash /root/catkin_ws/src/project_ppo/scripts/run_production_200k.sh"
    exit 1
fi

echo "[1/6] Cleaning up any existing processes..."
pkill -9 -f "gzserver|gzclient|gazebo|roscore|rosmaster|rosout|roslaunch|python3.*main.py" 2>/dev/null || true
sleep 3

echo "[2/6] Sourcing ROS environment..."
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
export TURTLEBOT3_MODEL=burger
export DISPLAY=:1

echo "[3/6] Starting roscore..."
nohup roscore > /tmp/roscore_prod.log 2>&1 &
sleep 3

# Wait for roscore
until rostopic list >/dev/null 2>&1; do 
    echo "  Waiting for roscore..."
    sleep 1
done
echo "  ✓ roscore ready"

echo "[4/6] Launching Small House with GUI..."
cd $PROJECT_DIR
nohup roslaunch project_ppo navbot_small_house.launch gui:=true > /tmp/gazebo_prod.log 2>&1 &
sleep 10

# Wait for Gazebo topics
TIMEOUT=60
ELAPSED=0
until rostopic list | grep -q "/scan" && rostopic list | grep -q "/clock" && rostopic list | grep -q "/odom"; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Gazebo topics not available after ${TIMEOUT}s"
        exit 1
    fi
    if [ $((ELAPSED % 10)) -eq 0 ]; then
        echo "  Waiting for Gazebo topics... (${ELAPSED}s)"
    fi
done
echo "  ✓ Gazebo ready (waited ${ELAPSED}s)"

# Verify processes
GZSERVER_PID=$(pgrep gzserver)
GZCLIENT_PID=$(pgrep gzclient)
echo "  ✓ gzserver PID: $GZSERVER_PID"
echo "  ✓ gzclient PID: $GZCLIENT_PID"

echo "[5/6] Starting production training..."
echo ""
echo "  Configuration:"
echo "    • Total timesteps: 200,000"
echo "    • Episode length: 800 steps"
echo "    • Iteration size: 8,000 steps"
echo "    • Total iterations: 25"
echo "    • Checkpoint interval: every 2 iterations"
echo "    • Vision: ResNet18 (frozen) + 64-dim projection"
echo "    • Sampler: distance-uniform, start clearance=0.50m, goal=0.30m"
echo "    • Run name: $RUN_NAME"
echo ""
echo "  Training will run in background."
echo "  You can close the Gazebo GUI window (gzclient) without stopping training."
echo ""
echo "  Monitor: docker exec navbot-ppo tail -f $LOG_FILE"
echo ""

cd $PROJECT_DIR/src

nohup python3 main.py \
    --mode train \
    --method_name "$RUN_NAME" \
    --vision_backbone resnet18 \
    --vision_proj_dim 64 \
    --timesteps_per_episode 800 \
    --steps_per_iteration 8000 \
    --max_timesteps 200000 \
    --use_map_sampler \
    --distance_uniform \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "  ✓ Training started (PID: $TRAIN_PID)"

sleep 5

# Verify training process started
if ps -p $TRAIN_PID > /dev/null; then
    echo "  ✓ Training process running"
else
    echo "  ✗ ERROR: Training process died immediately"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 "$LOG_FILE"
    exit 1
fi

echo "[6/6] Waiting for initial episodes..."
sleep 20

echo ""
echo "=========================================="
echo "Training Status (Initial)"
echo "=========================================="
echo ""

tail -30 "$LOG_FILE"

echo ""
echo "=========================================="
echo "Production Training Started Successfully!"
echo "=========================================="
echo ""
echo "📊 Status:"
echo "  • Training PID: $TRAIN_PID"
echo "  • Run name: $RUN_NAME"
echo "  • Output: $PROJECT_DIR/src/runs/$RUN_NAME/"
echo "  • Log file: $LOG_FILE"
echo ""
echo "🔍 Monitor progress:"
echo "  docker exec navbot-ppo tail -f $LOG_FILE"
echo ""
echo "📈 Check processes:"
echo "  docker exec navbot-ppo ps aux | grep -E 'python3|gzserver|gzclient'"
echo ""
echo "💾 Checkpoints saved to:"
echo "  $PROJECT_DIR/src/runs/$RUN_NAME/checkpoints/"
echo ""
echo "⚠️  You can close Gazebo GUI (gzclient) without stopping training."
echo ""
