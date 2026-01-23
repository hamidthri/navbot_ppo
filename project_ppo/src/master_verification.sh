#!/bin/bash
# Master verification script: Kill processes, Launch Gazebo+RViz, Run training
# Updated with clearance_goal=0.30m (above collision threshold 0.2m)

set -e

echo "========================================================================"
echo "MASTER VERIFICATION: Map Sampler (goal_clearance=0.30m) + Gazebo + RViz"
echo "========================================================================"
echo ""

# ============================================================================
# STEP 1: KILL ALL PROCESSES
# ============================================================================
echo "[1/5] Killing all ROS/Gazebo/training processes..."
pkill -9 -f gazebo || true
pkill -9 -f gzserver || true
pkill -9 -f gzclient || true
pkill -9 -f rosmaster || true
pkill -9 -f rosout || true
pkill -9 -f roslaunch || true
pkill -9 -f 'python.*main.py' || true
pkill -9 -f rviz || true
sleep 2
echo "✓ All processes killed"
echo ""

# ============================================================================
# STEP 2: SOURCE ROS ENVIRONMENT
# ============================================================================
echo "[2/5] Sourcing ROS environment..."
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
export TURTLEBOT3_MODEL=burger
export DISPLAY=:1
echo "✓ Environment configured (DISPLAY=$DISPLAY)"
echo ""

# ============================================================================
# STEP 3: LAUNCH GAZEBO WITH GUI
# ============================================================================
echo "[3/5] Launching Gazebo Small House world with GUI..."
roslaunch project navbot_small_house.launch gui:=true > /tmp/gazebo_output.log 2>&1 &
GAZEBO_PID=$!
echo "   Gazebo PID: $GAZEBO_PID"
echo "   Waiting 10 seconds for Gazebo to initialize..."
sleep 10

# Verify Gazebo is running
if ! ps -p $GAZEBO_PID > /dev/null; then
    echo "ERROR: Gazebo failed to start!"
    cat /tmp/gazebo_output.log
    exit 1
fi

# Verify topics
echo "   Verifying ROS topics..."
rostopic list | grep -E '(/scan|/odom|/camera)' || {
    echo "ERROR: Required topics not available!"
    rostopic list
    exit 1
}
echo "✓ Gazebo running with topics: /scan, /odom, /camera/rgb/image_raw"
echo ""

# ============================================================================
# STEP 4: LAUNCH RVIZ
# ============================================================================
echo "[4/5] Launching RViz with custom config..."
RVIZ_CONFIG="/root/catkin_ws/src/project_ppo/rviz/navbot_verification.rviz"

if [ -f "$RVIZ_CONFIG" ]; then
    rosrun rviz rviz -d "$RVIZ_CONFIG" > /tmp/rviz_output.log 2>&1 &
    RVIZ_PID=$!
    echo "   RViz PID: $RVIZ_PID"
    echo "   Config: $RVIZ_CONFIG"
else
    echo "   WARNING: RViz config not found, launching with default..."
    rosrun rviz rviz > /tmp/rviz_output.log 2>&1 &
    RVIZ_PID=$!
    echo "   RViz PID: $RVIZ_PID"
fi

sleep 3
echo "✓ RViz launched"
echo ""

# ============================================================================
# STEP 5: RUN SHORT TRAINING WITH MAP SAMPLER
# ============================================================================
echo "[5/5] Starting SHORT training run with map sampler..."
echo "   Configuration:"
echo "     - Method: verify_map_sampler_final"
echo "     - Episode steps: 50"
echo "     - Iteration steps: 400"
echo "     - Max timesteps: 800"
echo "     - Start clearance: 0.70m (safe)"
echo "     - Goal clearance: 0.30m (above 0.2m collision threshold)"
echo "     - Map sampler: ENABLED"
echo "     - Debug logs: ENABLED"
echo ""

cd /root/catkin_ws/src/project_ppo/src

python3 main.py \
    --method_name verify_map_sampler_final \
    --timesteps_per_episode 50 \
    --steps_per_iteration 400 \
    --max_timesteps 800 \
    --vision_backbone resnet18 \
    --vision_proj_dim 64 \
    --use_map_sampler \
    --debug_sampler 2>&1 | tee /tmp/training_verification.log

TRAIN_EXIT=$?

echo ""
echo "========================================================================"
echo "VERIFICATION COMPLETE"
echo "========================================================================"
echo ""
echo "Exit codes:"
echo "  Training: $TRAIN_EXIT"
echo "  Gazebo PID: $GAZEBO_PID (running: $(ps -p $GAZEBO_PID > /dev/null && echo 'YES' || echo 'NO'))"
echo "  RViz PID: $RVIZ_PID (running: $(ps -p $RVIZ_PID > /dev/null && echo 'YES' || echo 'NO'))"
echo ""
echo "Logs:"
echo "  Training: /tmp/training_verification.log"
echo "  Gazebo: /tmp/gazebo_output.log"
echo "  RViz: /tmp/rviz_output.log"
echo ""

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Training completed successfully!"
    
    # Extract key metrics
    echo ""
    echo "Key metrics from training:"
    grep -E '\[RESET\]' /tmp/training_verification.log | head -10 || echo "  (no reset logs found)"
    echo ""
    grep -E 'Entropy=' /tmp/training_verification.log | tail -3 || echo "  (no entropy logs found)"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT"
    echo ""
    echo "Last 30 lines of training log:"
    tail -30 /tmp/training_verification.log
fi

echo ""
echo "========================================================================"
echo "PROCESSES REMAIN RUNNING (Gazebo + RViz)"
echo "To kill: pkill -9 -f 'gazebo|rviz|rosmaster'"
echo "========================================================================"
