#!/bin/bash
# Monitor SAC training progress

echo "=== SAC Training Monitor ==="
docker exec navbot-ppo bash -c "tail -f /root/catkin_ws/src/project_ppo/src/logs/sac_*.log"
