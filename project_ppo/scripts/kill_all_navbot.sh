#!/usr/bin/env bash
trap '' TERM INT  # Ignore signals

CONTAINER="${1:-navbot-ppo}"
MAX_ATTEMPTS=5

echo "================================================================================"
echo "KILL SCRIPT: ROS/Gazebo/Training"
echo "Container: ${CONTAINER}"
echo "================================================================================"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}" > /dev/null 2>&1; then
  echo "[ERROR] Container '${CONTAINER}' is not running."
  exit 1
fi

get_pids() {
  # Use exact process names (not -f) to avoid matching command arguments
  docker exec "${CONTAINER}" bash -c "
    {
      pgrep '^roslaunch$' 2>/dev/null;
      pgrep '^roscore$' 2>/dev/null;
      pgrep '^rosmaster$' 2>/dev/null;
      pgrep '^rosout$' 2>/dev/null;
      pgrep '^gzserver$' 2>/dev/null;
      pgrep '^gzclient$' 2>/dev/null;
      pgrep '^rostopic$' 2>/dev/null;
      pgrep '^rosservice$' 2>/dev/null;
      pgrep '^rosnode$' 2>/dev/null;
      # For python processes, match the full command line but filter
      ps -eo pid,args | awk '/[p]ython.*main\.py/ {print \$1}';
    } | sort -u | tr '\n' ' '
  " 2>/dev/null | xargs
}

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  PIDS=$(get_pids)
  
  if [[ -z "${PIDS}" ]]; then
    echo "✓ All processes cleared!"
    exit 0
  fi
  
  echo ""
  echo "[Attempt ${attempt}/${MAX_ATTEMPTS}] PIDs: ${PIDS}"
  
  if [[ "${attempt}" -le 2 ]]; then
    echo "  → SIGTERM..."
    docker exec "${CONTAINER}" bash -c "kill -15 ${PIDS} 2>/dev/null || true"
  else
    echo "  → SIGKILL..."
    docker exec "${CONTAINER}" bash -c "kill -9 ${PIDS} 2>/dev/null || true"
  fi
  
  sleep 2
done

# Final check
FINAL_PIDS=$(get_pids)

if [[ -z "${FINAL_PIDS}" ]]; then
  echo "✓ All cleared!"
  exit 0
fi

echo ""
echo "⚠ WARNING: Processes still alive: ${FINAL_PIDS}"
docker exec "${CONTAINER}" bash -c "ps -p ${FINAL_PIDS} 2>/dev/null || true"
echo ""
echo "Last resort: docker restart ${CONTAINER}"
exit 1
