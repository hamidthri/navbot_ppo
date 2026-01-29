# LiDAR SAC Training

This folder contains the LiDAR-only SAC implementation for baseline comparison.

## Quick Start

### Run Training (Automated)
```bash
# Inside Docker container:
bash /root/catkin_ws/src/project_ppo/src/sac_lidar/run_lidar_training.sh

# Or from host:
docker exec -it navbot-ppo bash /root/catkin_ws/src/project_ppo/src/sac_lidar/run_lidar_training.sh
```

The script automatically:
1. ✓ Kills existing processes
2. ✓ Launches Gazebo with small house world
3. ✓ Starts LiDAR SAC training
4. ✓ Logs output to timestamped file in `/tmp/`

### Training Configuration
- **Timesteps**: 10,000 (configurable in `train_sac_lidar.py`)
- **State**: 16D LiDAR scans only
- **Replay Buffer**: 1M capacity (memory-efficient without images)
- **Batch Size**: 256
- **Hidden Dim**: 256

### Files
- `run_lidar_training.sh` - Automated training launcher
- `train_sac_lidar.py` - Main training script
- `sac.py` - SAC agent
- `sac_networks.py` - Actor/Critic networks
- `replay_buffer.py` - Standard replay buffer
- `environment_small_house.py` - LiDAR-only environment
- `small_house_region_sampler.py` - Goal/spawn sampling

### Monitor Training
```bash
# Watch live training output
docker exec navbot-ppo tail -f /tmp/lidar_training_*.log

# Check latest log
docker exec navbot-ppo ls -lt /tmp/lidar_training_*.log | head -1
```

### Evaluate Trained Model
```bash
# Inside Docker container
cd /root/catkin_ws/src/project_ppo/src/sac_lidar

# Evaluate with default settings (10 episodes)
python3 eval_sac_lidar.py --model models/sac_lidar_10k/sac_lidar_final_10001.pth

# Evaluate with custom episodes
python3 eval_sac_lidar.py --model models/sac_lidar_10k/sac_lidar_5000.pth --episodes 20

# From host
docker exec -it navbot-ppo bash -c "cd /root/catkin_ws/src/project_ppo/src/sac_lidar && python3 eval_sac_lidar.py --model models/sac_lidar_10k/sac_lidar_final_10001.pth"
```

**Note**: Gazebo must be running before evaluation. The script will use the existing Gazebo instance.

### Models
Saved to: `models/sac_lidar_10k/`
- Checkpoints every 5000 steps
- Final model after 10k steps

## Architecture

```
LiDAR Scan (16D)
      ↓
  Actor Network (256 → 256 → action)
      ↓
  Critic Network (256 → 256 → Q-value)
```

## Comparison with Vision
- **Faster**: No image processing overhead
- **Memory Efficient**: Larger replay buffer (1M vs 50k)
- **Larger Batch**: 256 vs 64
- **Baseline**: Pure LiDAR performance without visual information

## Notes
- LiDAR-only baseline for comparison with vision approaches
- Faster training due to simpler state representation
- Good for benchmarking vision improvements
- Success rate and collision metrics tracked during episodes
