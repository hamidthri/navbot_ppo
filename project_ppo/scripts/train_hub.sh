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
#   # Using environment variables
#   METHOD_NAME=my_run MAX_TIMESTEPS=50000 ./train_hub.sh
#
################################################################################

set -e  # Exit on error

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
            DISTANCE_UNIFORM="$2"
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
        --runs_root)
            RUNS_ROOT="$2"
            shift 2
            ;;
        --use_vision)
            USE_VISION="$2"
            shift 2
            ;;
        --vision_model)
            VISION_MODEL="$2"
            shift 2
            ;;
        --vision_dim)
            VISION_DIM="$2"
            shift 2
            ;;
        --help|-h)
            echo "Training Hub Launcher"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --method_name <name>           Training run name (default: fuzzy3_training)"
            echo "  --reward_type <type>           Reward function: fuzzy3|legacy (default: fuzzy3)"
            echo "  --sampler_mode <mode>          Sampler: rect_regions|legacy (default: rect_regions)"
            echo "  --distance_uniform <bool>      Use shaped distance sampling (default: true)"
            echo "  --max_timesteps <int>          Total timesteps (default: 200000)"
            echo "  --timesteps_per_episode <int>  Episode length (default: 100)"
            echo "  --steps_per_iteration <int>    PPO update frequency (default: 2048)"
            echo "  --runs_root <path>             Output directory (default: runs)"
            echo "  --use_vision <bool>            Enable vision encoder (default: false)"
            echo "  --vision_model <model>         Vision backbone: resnet18|dinov2 (default: resnet18)"
            echo "  --vision_dim <int>             Vision feature dimension (default: 512)"
            echo "  --help, -h                     Show this help message"
            echo ""
            echo "Environment Variables (can also be used):"
            echo "  METHOD_NAME, REWARD_TYPE, MAX_TIMESTEPS, etc."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the python command
PYTHON_CMD="python3 main.py \
  --mode train \
  --method_name ${METHOD_NAME} \
  --reward_type ${REWARD_TYPE} \
  --sampler_mode ${SAMPLER_MODE} \
  --max_timesteps ${MAX_TIMESTEPS} \
  --timesteps_per_episode ${TIMESTEPS_PER_EPISODE} \
  --steps_per_iteration ${STEPS_PER_ITERATION} \
  --runs_root ${RUNS_ROOT}"

# Add distance_uniform flag if enabled
if [[ "${DISTANCE_UNIFORM,,}" == "true" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --distance_uniform"
fi

# Add vision flags if enabled
if [[ "${USE_VISION,,}" == "true" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --use_vision --vision_model ${VISION_MODEL} --vision_dim ${VISION_DIM}"
fi

# Print configuration
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
echo "Runs Root:            ${RUNS_ROOT}"
echo "Use Vision:           ${USE_VISION}"
if [[ "${USE_VISION,,}" == "true" ]]; then
    echo "Vision Model:         ${VISION_MODEL}"
    echo "Vision Dim:           ${VISION_DIM}"
fi
echo "================================================================================"
echo ""
echo "Full command:"
echo "${PYTHON_CMD}"
echo ""
echo "================================================================================"

# Execute inside Docker container
docker exec navbot-ppo bash -lc "
    source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    cd /root/catkin_ws/src/project_ppo/src && \
    ${PYTHON_CMD}
"

# Check exit code
EXIT_CODE=$?
echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: ${RUNS_ROOT}/${METHOD_NAME}/"
else
    echo "Training failed with exit code: ${EXIT_CODE}"
fi
echo "================================================================================"

exit $EXIT_CODE
