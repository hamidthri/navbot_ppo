# SAC LiDAR Navigation

LiDAR-only navigation using Soft Actor-Critic (SAC). Baseline implementation for comparison with vision-based approaches.

## Running with Docker

```bash
# 1. Start container
docker compose up -d
docker exec -it navbot-ppo bash

# 2. Navigate to sac_lidar folder
cd /workspace/project_ppo/src/sac_lidar

# 3. Run training with shell script (recommended)
./run_lidar_training.sh --max_timesteps 200000 --reward_type legacy

# 4. Or run training directly with Python
python3 train_sac_lidar.py \
    --max_timesteps 200000 \
    --reward_type legacy \
    --run_name my_training_run
```

## Running with ROS1 (Native Install)

```bash
# 1. Launch Gazebo (in one terminal)
export TURTLEBOT3_MODEL=burger
roslaunch project_ppo navbot_small_house.launch gui:=false

# 2. Run training with shell script (in another terminal)
cd project_ppo/src/sac_lidar
./run_lidar_training.sh --max_timesteps 200000

# 3. Or run training directly with Python
cd project_ppo/src/sac_lidar
python3 train_sac_lidar.py \
    --max_timesteps 200000 \
    --reward_type legacy
```

## Training Arguments

- `--max_timesteps`: Total training steps - default: `200000`
- `--reward_type`: Reward function (`legacy`, `lyapunov`) - default: `legacy`
- `--run_name`: Custom name for this training run (optional)
- `--resume`: Path to checkpoint to resume training (optional)

## Evaluation

```bash
# With Gazebo running:
python3 eval_sac_lidar.py \
    --model models/sac_lidar_200k/sac_lidar_final_200001.pth \
    --episodes 10

# Sequential navigation evaluation:
python3 eval_sequential_r6_r7_r8_r11_r13.py \
    --model_path models/sac_lidar_200k/sac_lidar_final_200001.pth
```

## Configuration

Edit `config_lidar.yaml` to change hyperparameters (learning rates, batch size, buffer size, network architecture, etc.).
