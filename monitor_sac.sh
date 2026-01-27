#!/bin/bash
# Monitor SAC training progress

echo "======================================"
echo "SAC TRAINING MONITOR"
echo "======================================"
echo ""

# Check if processes are running
echo "[1] Process Status:"
docker exec navbot-ppo bash -c "ps aux | grep -E 'gzserver|python3.*sac' | grep -v grep | wc -l" | \
  xargs -I {} echo "  Running processes: {}"

# Show latest training output
echo ""
echo "[2] Latest Training Output:"
docker exec navbot-ppo bash -c "tail -20 /root/catkin_ws/sac_train.log"

echo ""
echo "======================================"
echo "To monitor live: docker exec navbot-ppo tail -f /root/catkin_ws/sac_train.log"
echo "To stop training: docker exec navbot-ppo pkill -f 'python3.*sac'"
echo "======================================"
