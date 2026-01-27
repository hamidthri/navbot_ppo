#!/bin/bash
################################################################################
# Training Hub Launcher - Single Source of Truth for PPO Training
#
# This is the ONLY script for launching training runs.
# All training parameters can be overridden via environment variables or CLI args.
#
# Usage:
#   ./train_hub.sh [OPTIONS]
#
# Examples:
#   # Quick test run
#   ./train_hub.sh --method_name test_run --max_timesteps 1000
#
#   # Full production run with fuzzy3 reward
#   ./train_hub.sh --method_name prod_fuzzy3_200k \
#                  --reward_type fuzzy3 \
#                  --max_timesteps 200000
#
#   # With Gazebo (headless)
#   ./train_hub.sh --method_name my_run --with_gazebo
#
#   # With Gazebo GUI
#   ./train_hub.sh --method_name my_run --with_gazebo --gazebo_gui
#
#   # Using environment variables
#   METHOD_NAME=my_run MAX_TIMESTEPS=50000 ./train_hub.sh
#
################################################################################

# Default parameters (production-ready settings)
METHOD_NAME="${METHOD_NAME:-fuzzy3_training}"
REWARD_TYPE="${REWARD_TYPE:-fuzzy3}"
SAMPLER_MODE="${SAMPLER_MODE:-rect_regions}"
DISTANCE_UNIFORM="${DISTANCE_UNIFORM:-true}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-200000}"
TIMESTEPS_PER_EPISODE="${TIMESTEPS_PER_EPISODE:-100}"
STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-2048}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
USE_VISION="${USE_VISION:-false}"
VISION_MODEL="${VISION_MODEL:-resnet18}"
VISION_DIM="${VISION_DIM:-512}"
ARCHITECTURE="${ARCHITECTURE:-default}"
SAVE_EVERY_ITERATIONS="${SAVE_EVERY_ITERATIONS:-2}"

# Gazebo/ROS management parameters
WITH_GAZEBO="${WITH_GAZEBO:-false}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
MAP="${MAP:-small_house}"
NO_KILL="${NO_KILL:-false}"
TIMEOUT_READY_SEC="${TIMEOUT_READY_SEC:-60}"

# Curriculum parameters (passed through to Python)
CURRICULUM_MIN_DIST=""
CURRICULUM_MAX_DIST=""
CURRICULUM_STEPS=""

# Parse command-line arguments (override defaults)
while [[ $# -gt 0 ]]; do
    case $1 in
        --method_name)
            METHOD_NAME="$2"
            shift 2
            ;;
        --reward_type)
            REWARD_TYPE="$2"
            shift 2
            ;;
        --sampler_mode)
            SAMPLER_MODE="$2"
            shift 2
            ;;
        --distance_uniform)
            if [[ "$2" == "true" ]] || [[ "$2" == "false" ]]; then
                DISTANCE_UNIFORM="$2"
                shift 2
            else
                DISTANCE_UNIFORM="true"
                shift 1
            fi
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
        --runs_root)
            RUNS_ROOT="$2"
            shift 2
            ;;
        --save_every_iterations)
            SAVE_EVERY_ITERATIONS="$2"
            shift 2
            ;;
        --use_vision)
            USE_VISION="$2"
            shift 2
            ;;
        --vision_model|--vision_backbone)
            VISION_MODEL="$2"
            shift 2
            ;;
        --vision_dim|--vision_proj_dim)
            VISION_DIM="$2"
            shift 2
            ;;
        --architecture)
            ARCHITECTURE="$2"
            shift 2
            ;;
        --with_gazebo)
            WITH_GAZEBO="true"
            shift 1
            ;;
        --gazebo_gui)
            GAZEBO_GUI="true"
            WITH_GAZEBO="true"  # Implies with_gazebo
            shift 1
            ;;
        --map)
            MAP="$2"
            shift 2
            ;;
        --no_kill)
            NO_KILL="true"
            shift 1
            ;;
        --timeout_ready_sec)
            TIMEOUT_READY_SEC="$2"
            shift 2
            ;;
        --curriculum_min_dist)
            CURRICULUM_MIN_DIST="$2"
            shift 2
            ;;
        --curriculum_max_dist)
            CURRICULUM_MAX_DIST="$2"
            shift 2
            ;;
        --curriculum_steps)
            CURRICULUM_STEPS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Training Hub Launcher"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Training Options:"
            echo "  --method_name <name>           Training run name (default: fuzzy3_training)"
            echo "  --reward_type <type>           Reward function: fuzzy3|fuzzy3_v4|legacy|potential_field (default: fuzzy3)"
            echo "  --sampler_mode <mode>          Sampler: rect_regions|legacy (default: rect_regions)"
            echo "  --distance_uniform [bool]      Use shaped distance sampling (default: true)"
            echo "  --max_timesteps <int>          Total timesteps (default: 200000)"
            echo "  --timesteps_per_episode <int>  Episode length (default: 100)"
            echo "  --steps_per_iteration <int>    PPO update frequency (default: 2048)"
            echo "  --runs_root <path>             Output directory (default: runs)"
            echo "  --save_every_iterations <int>  Checkpoint frequency (default: 2)"
            echo "  --use_vision <bool>            Enable vision encoder (default: false)"
            echo "  --vision_model <model>         Vision backbone: resnet18|dinov2 (default: resnet18)"
            echo "  --vision_dim <int>             Vision feature dimension (default: 512)"
            echo ""
            echo "ROS/Gazebo Management Options:"
            echo "  --with_gazebo                  Launch ROS + Gazebo (default: false)"
            echo "  --gazebo_gui                   Launch Gazebo with GUI (implies --with_gazebo)"
            echo "  --map <name>                   Map/world to use (default: small_house)"
            echo "  --no_kill                      Don't kill ROS/Gazebo on exit (default: false)"
            echo "  --timeout_ready_sec <int>      Timeout for ROS/Gazebo ready (default: 60)"
            echo ""
            echo "  --help, -h                     Show this help message"
            echo ""
            echo "Environment Variables (can also be used):"
            echo "  METHOD_NAME, REWARD_TYPE, MAX_TIMESTEPS, WITH_GAZEBO, etc."
            echo ""
            echo "Examples:"
            echo "  # Quick test with auto-launched Gazebo"
            echo "  ./train_hub.sh --method_name test --max_timesteps 1000 --with_gazebo"
            echo ""
            echo "  # Production run with GUI"
            echo "  ./train_hub.sh --method_name prod_run --gazebo_gui --max_timesteps 200000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================

# Cleanup function for trap
cleanup() {
    local exit_code=$?
    echo ""
    echo "================================================================================"
    echo "Cleanup triggered (exit code: ${exit_code})"
    echo "================================================================================"
    
    if [[ "${NO_KILL}" == "true" ]]; then
        echo "NO_KILL flag set - leaving ROS/Gazebo running"
    else
        if [[ "${WITH_GAZEBO}" == "true" ]]; then
            echo "Stopping ROS/Gazebo processes..."
            docker exec navbot-ppo bash -c "pkill -9 gzserver 2>/dev/null; pkill -9 gzclient 2>/dev/null; pkill -9 rosmaster 2>/dev/null; true"
            echo "ROS/Gazebo processes stopped"
        fi
    fi
    
    echo "Cleanup complete"
    exit ${exit_code}
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# ============================================================================
# ROS/Gazebo Management
# ============================================================================

if [[ "${WITH_GAZEBO}" == "true" ]]; then
    echo "================================================================================"
    echo "ROS/Gazebo Launch Requested"
    echo "================================================================================"
    echo "Map:                  ${MAP}"
    echo "GUI:                  ${GAZEBO_GUI}"
    echo "Timeout:              ${TIMEOUT_READY_SEC}s"
    echo "Kill on exit:         $([ "${NO_KILL}" == "true" ] && echo "NO" || echo "YES")"
    echo "================================================================================"
    echo ""
    
    # Check DISPLAY for GUI mode
    if [[ "${GAZEBO_GUI}" == "true" ]]; then
        DISPLAY_VAR=$(docker exec navbot-ppo bash -c "echo \$DISPLAY" 2>/dev/null || echo "")
        if [[ -z "${DISPLAY_VAR}" ]]; then
            echo "WARNING: DISPLAY not set in container - GUI may not work"
            echo "  To enable GUI, run container with: -e DISPLAY=:1 -v /tmp/.X11-unix:/tmp/.X11-unix"
            echo "  Falling back to headless mode..."
            GAZEBO_GUI="false"
        else
            echo "✓ DISPLAY set to: ${DISPLAY_VAR}"
        fi
    fi
    
    echo ""
    echo "[1/4] Cleaning up any existing ROS/Gazebo processes..."
    docker exec navbot-ppo bash -c "pkill -9 python3 2>/dev/null; pkill -9 gzserver 2>/dev/null; pkill -9 gzclient 2>/dev/null; pkill -9 rosmaster 2>/dev/null; pkill -9 rosout 2>/dev/null; true"
    sleep 3
    echo "  ✓ Cleanup complete"
    
    echo ""
    echo "[2/4] Starting roscore..."
    docker exec navbot-ppo bash -lc "
        source /opt/ros/noetic/setup.bash && \
        nohup roscore > /tmp/roscore_hub.log 2>&1 &
    " &
    sleep 3
    
    # Wait for roscore
    echo "  Waiting for roscore to be ready..."
    ROSCORE_READY=false
    for i in $(seq 1 ${TIMEOUT_READY_SEC}); do
        if docker exec navbot-ppo bash -lc "source /opt/ros/noetic/setup.bash && timeout 2 rostopic list >/dev/null 2>&1"; then
            ROSCORE_READY=true
            echo "  ✓ roscore ready (${i}s)"
            break
        fi
        sleep 1
        if (( i % 5 == 0 )); then
            echo "    Still waiting... (${i}s)"
        fi
    done
    
    if [[ "${ROSCORE_READY}" == "false" ]]; then
        echo "  ✗ ERROR: roscore failed to start within ${TIMEOUT_READY_SEC}s"
        echo ""
        echo "Diagnostics:"
        docker exec navbot-ppo bash -c "tail -50 /tmp/roscore_hub.log 2>/dev/null || echo 'No roscore log'"
        exit 1
    fi
    
    echo ""
    echo "[3/4] Launching Gazebo (${MAP})..."
    GUI_ARG="false"
    if [[ "${GAZEBO_GUI}" == "true" ]]; then
        GUI_ARG="true"
    fi
    
    # Determine launch file based on MAP variable
    LAUNCH_FILE="navbot_small_house.launch"
    if [[ "${MAP}" == "stage_1" ]]; then
        LAUNCH_FILE="navbot_stage_1.launch"
    elif [[ "${MAP}" == "small_house" ]]; then
        LAUNCH_FILE="navbot_small_house.launch"
    fi
    
    docker exec navbot-ppo bash -lc "
        source /opt/ros/noetic/setup.bash && \
        source /root/catkin_ws/devel/setup.bash && \
        export TURTLEBOT3_MODEL=burger && \
        cd /root/catkin_ws/src/project_ppo && \
        nohup roslaunch project_ppo ${LAUNCH_FILE} gui:=${GUI_ARG} > /tmp/gazebo_hub.log 2>&1 &
    " &
    sleep 5
    
    echo "  Waiting for Gazebo topics to be ready..."
    GAZEBO_READY=false
    for i in $(seq 1 ${TIMEOUT_READY_SEC}); do
        SCAN_OK=$(docker exec navbot-ppo bash -lc "source /opt/ros/noetic/setup.bash && timeout 2 rostopic list 2>/dev/null | grep -c '/scan'" || echo "0")
        ODOM_OK=$(docker exec navbot-ppo bash -lc "source /opt/ros/noetic/setup.bash && timeout 2 rostopic list 2>/dev/null | grep -c '/odom'" || echo "0")
        # Sanitize output to ensure single numeric value
        SCAN_OK=$(echo "${SCAN_OK}" | tail -1 | tr -d '\n\r ')
        ODOM_OK=$(echo "${ODOM_OK}" | tail -1 | tr -d '\n\r ')
        
        if [[ "${SCAN_OK}" -gt 0 ]] 2>/dev/null && [[ "${ODOM_OK}" -gt 0 ]] 2>/dev/null; then
            GAZEBO_READY=true
            echo "  ✓ Gazebo ready - /scan and /odom topics available (${i}s)"
            break
        fi
        sleep 1
        if (( i % 10 == 0 )); then
            echo "    Still waiting for topics... (${i}s)"
        fi
    done
    
    if [[ "${GAZEBO_READY}" == "false" ]]; then
        echo "  ✗ ERROR: Gazebo topics not available within ${TIMEOUT_READY_SEC}s"
        echo ""
        echo "Diagnostics:"
        echo "--- rostopic list ---"
        docker exec navbot-ppo bash -lc "source /opt/ros/noetic/setup.bash && rostopic list 2>&1 | head -20 || echo 'Failed'"
        echo "--- gazebo log ---"
        docker exec navbot-ppo bash -c "tail -100 /tmp/gazebo_hub.log 2>/dev/null || echo 'No gazebo log'"
        echo "--- processes ---"
        docker exec navbot-ppo bash -c "ps aux | grep -E 'gzserver|gzclient' | grep -v grep || echo 'No gazebo processes'"
        exit 1
    fi
    
    echo ""
    echo "[4/4] Verifying Gazebo services..."
    SERVICES_OK=$(docker exec navbot-ppo bash -lc "source /opt/ros/noetic/setup.bash && timeout 5 rosservice list 2>/dev/null | grep -c '/gazebo/'" || echo "0")
    SERVICES_OK=$(echo "${SERVICES_OK}" | tail -1 | tr -d '\n\r ')
    if [[ "${SERVICES_OK}" -gt 0 ]] 2>/dev/null; then
        echo "  ✓ Gazebo services available"
    else
        echo "  ⚠ WARNING: Gazebo services not detected (training may fail)"
    fi
    
    echo ""
    echo "================================================================================"
    echo "ROS/Gazebo Ready!"
    echo "================================================================================"
    sleep 2
fi

# ============================================================================
# Camera Readiness Check (if vision training)
# ============================================================================

if [[ "${USE_VISION,,}" == "true" ]]; then
    echo ""
    echo "================================================================================"
    echo "Camera Readiness Check (Vision Training Required)"
    echo "================================================================================"
    echo "Vision Enabled:       YES"
    echo "Expected Topic:       /robot_camera/image_raw"
    echo ""
    
    echo "[1/3] Checking if camera topic exists..."
    CAMERA_TOPIC_EXISTS=$(docker exec navbot-ppo bash -lc "source /opt/ros/noetic/setup.bash && timeout 5 rostopic list 2>/dev/null | grep -c '/robot_camera/image_raw'" || echo "0")
    CAMERA_TOPIC_EXISTS=$(echo "${CAMERA_TOPIC_EXISTS}" | tail -1 | tr -d '\n\r ')
    
    if [[ "${CAMERA_TOPIC_EXISTS}" -gt 0 ]] 2>/dev/null; then
        echo "  ✓ Camera topic /robot_camera/image_raw exists"
    else
        echo "  ✗ ERROR: Camera topic /robot_camera/image_raw not found"
        echo ""
        echo "Available topics:"
        docker exec navbot-ppo bash -lc "source /opt/ros/noetic/setup.bash && rostopic list 2>/dev/null | head -20 || echo 'Failed to list topics'"
        echo ""
        echo "ERROR: Vision training requires camera topic /robot_camera/image_raw"
        echo "       Ensure Gazebo is running with robot camera sensor enabled."
        exit 1
    fi
    
    echo "[2/3] Checking camera publish frequency..."
    CAMERA_HZ=$(docker exec navbot-ppo bash -lc "source /opt/ros/noetic/setup.bash && timeout 3 rostopic hz /robot_camera/image_raw 2>&1 | grep 'average rate' | awk '{print \$3}' || echo '0'")
    
    if [[ -z "${CAMERA_HZ}" ]] || [[ "${CAMERA_HZ}" == "0" ]]; then
        echo "  ⚠ WARNING: Could not determine camera frequency (may still work if camera is slow)"
    else
        echo "  ✓ Camera publishing at ~${CAMERA_HZ} Hz"
    fi
    
    echo "[3/3] Verifying camera frames are being received..."
    CAMERA_FRAME_CHECK=$(docker exec navbot-ppo bash -lc "source /opt/ros/noetic/setup.bash && timeout 5 rostopic echo -n1 /robot_camera/image_raw/header/seq 2>/dev/null || echo 'FAIL'")
    
    if [[ "${CAMERA_FRAME_CHECK}" != "FAIL" ]] && [[ -n "${CAMERA_FRAME_CHECK}" ]]; then
        echo "  ✓ Camera frames confirmed (sequence: ${CAMERA_FRAME_CHECK})"
    else
        echo "  ✗ ERROR: No camera frames received within timeout"
        echo ""
        echo "Diagnostics:"
        echo "--- Camera info topic ---"
        docker exec navbot-ppo bash -lc "source /opt/ros/noetic/setup.bash && timeout 2 rostopic info /robot_camera/image_raw 2>/dev/null || echo 'Topic info unavailable'"
        echo "--- Gazebo processes ---"
        docker exec navbot-ppo bash -c "ps aux | grep -E 'gzserver|gzclient' | grep -v grep || echo 'No Gazebo processes'"
        echo ""
        echo "ERROR: Vision training requires active camera stream."
        echo "       Camera topic exists but no frames are being published."
        exit 1
    fi
    
    echo ""
    echo "================================================================================"
    echo "✓ Camera Ready for Vision Training"
    echo "================================================================================"
    sleep 1
fi

# ============================================================================
# Build Training Command
# ============================================================================

# Build the python command
PYTHON_CMD="python3 main.py \
  --mode train \
  --method_name ${METHOD_NAME} \
  --reward_type ${REWARD_TYPE} \
  --sampler_mode ${SAMPLER_MODE} \
  --max_timesteps ${MAX_TIMESTEPS} \
  --timesteps_per_episode ${TIMESTEPS_PER_EPISODE} \
  --steps_per_iteration ${STEPS_PER_ITERATION} \
  --save_every_iterations ${SAVE_EVERY_ITERATIONS} \
  --runs_root ${RUNS_ROOT}"

# Add distance_uniform flag if enabled
if [[ "${DISTANCE_UNIFORM,,}" == "true" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --distance_uniform"
fi

# Add vision flags if enabled
if [[ "${USE_VISION,,}" == "true" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --vision_backbone ${VISION_MODEL} --vision_proj_dim ${VISION_DIM} --architecture ${ARCHITECTURE}"
else
    # Explicitly disable vision by passing 'none' as backbone
    PYTHON_CMD="${PYTHON_CMD} --vision_backbone none"
fi

# Add curriculum flags if provided
if [[ -n "${CURRICULUM_MIN_DIST}" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --curriculum_min_dist ${CURRICULUM_MIN_DIST}"
fi
if [[ -n "${CURRICULUM_MAX_DIST}" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --curriculum_max_dist ${CURRICULUM_MAX_DIST}"
fi
if [[ -n "${CURRICULUM_STEPS}" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --curriculum_steps ${CURRICULUM_STEPS}"
fi

# ============================================================================
# Print Configuration and Execute
# ============================================================================

# Create log directory
LOG_DIR="/root/catkin_ws/${RUNS_ROOT}/${METHOD_NAME}/logs"
docker exec navbot-ppo bash -c "mkdir -p ${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_hub.log"

# Print configuration
echo ""
echo "================================================================================"
echo "Training Hub Launcher - Starting Training"
echo "================================================================================"
echo "Method Name:          ${METHOD_NAME}"
echo "Reward Type:          ${REWARD_TYPE}"
echo "Sampler Mode:         ${SAMPLER_MODE}"
echo "Distance Uniform:     ${DISTANCE_UNIFORM}"
echo "Max Timesteps:        ${MAX_TIMESTEPS}"
echo "Timesteps/Episode:    ${TIMESTEPS_PER_EPISODE}"
echo "Steps/Iteration:      ${STEPS_PER_ITERATION}"
echo "Save Every:           ${SAVE_EVERY_ITERATIONS} iterations"
echo "Runs Root:            ${RUNS_ROOT}"
echo "Use Vision:           ${USE_VISION}"
if [[ "${USE_VISION,,}" == "true" ]]; then
    echo "Vision Model:         ${VISION_MODEL}"
    echo "Vision Dim:           ${VISION_DIM}"
fi
echo ""
echo "ROS/Gazebo:           ${WITH_GAZEBO}"
if [[ "${WITH_GAZEBO}" == "true" ]]; then
    echo "Gazebo GUI:           ${GAZEBO_GUI}"
    echo "Map:                  ${MAP}"
fi
echo ""
echo "Log File:             ${LOG_FILE}"
echo "================================================================================"
echo ""
echo "Full command:"
echo "${PYTHON_CMD}"
echo ""
echo "================================================================================"

# Execute inside Docker container with logging
echo "Starting training... (check ${LOG_FILE} for details)"
echo ""

docker exec navbot-ppo bash -lc "
    source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    cd /root/catkin_ws/src/project_ppo/src && \
    ${PYTHON_CMD} 2>&1 | tee ${LOG_FILE}
"

# Check exit code
EXIT_CODE=$?
echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo "Results saved to: ${RUNS_ROOT}/${METHOD_NAME}/"
    echo "Log file: ${LOG_FILE}"
    
    # List created artifacts
    echo ""
    echo "Created artifacts:"
    docker exec navbot-ppo bash -c "
        cd /root/catkin_ws/${RUNS_ROOT}/${METHOD_NAME} && \
        echo '  Config:' && ls -lh *.yml *.yaml config.* 2>/dev/null | awk '{print \"    \" \$NF}' || echo '    (none)' && \
        echo '  Checkpoints:' && ls -lh checkpoints/*.pth 2>/dev/null | wc -l | awk '{print \"    \" \$1 \" files\"}' && \
        echo '  Logs:' && ls -lh logs/ 2>/dev/null | tail -n +2 | awk '{print \"    \" \$NF}' || echo '    (none)' && \
        echo '  Tensorboard:' && ls -lh tb/ 2>/dev/null | tail -n +2 | wc -l | awk '{print \"    \" \$1 \" files\"}' || echo '    (none)'
    "
else
    echo "✗ Training failed with exit code: ${EXIT_CODE}"
    echo "Check log file: ${LOG_FILE}"
    echo "Last 50 lines of log:"
    docker exec navbot-ppo bash -c "tail -50 ${LOG_FILE} 2>/dev/null || echo 'No log available'"
fi
echo "================================================================================"

exit $EXIT_CODE
