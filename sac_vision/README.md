# SAC Vision Navigation

Vision-based navigation using RGB camera + LiDAR fusion with Soft Actor-Critic (SAC).

## Prerequisites

This implementation requires the **AWS RoboMaker Small House World** for Gazebo simulation. The world should be cloned in the workspace root:

```bash
cd /path/to/navbot_ppo
git clone https://github.com/aws-robotics/aws-robomaker-small-house-world.git
```

For Docker users, this should already be set up in the container.

## Running with Docker

```bash
# 1. Start container
docker compose up -d
docker exec -it navbot-ppo bash

# 2. Navigate to sac_vision folder
cd /root/catkin_ws/src/sac_vision

# 3. Run training with shell script (recommended)
./run_vision_training.sh --fusion_type film --max_timesteps 200000 --reward_type legacy

# 4. Or run training directly with Python
python3 sac_training_vision.py \
    --fusion_type film \
    --max_timesteps 200000 \
    --reward_type legacy \
    --run_name my_training_run
```

## Running with ROS1 (Native Install)

```bash
# 1. Launch Gazebo (in one terminal)
export TURTLEBOT3_MODEL=burger
roslaunch project sac_small_house.launch gui:=false

# 2. Run training with shell script (in another terminal)
cd /path/to/navbot_ppo/sac_vision
./run_vision_training.sh --fusion_type film --max_timesteps 200000

# 3. Or run training directly with Python
cd /path/to/navbot_ppo/sac_vision
python3 sac_training_vision.py \
    --fusion_type film \
    --max_timesteps 200000 \
    --reward_type legacy
```

## Training Arguments

- `--fusion_type`: Fusion method (`film`, `concat`, `attention`, `gated`) - default: `film`
- `--max_timesteps`: Total training steps - default: `200000`
- `--reward_type`: Reward function (`legacy`, `lyapunov`) - default: `legacy`
- `--run_name`: Custom name for this training run (optional)
- `--resume`: Path to checkpoint to resume training (optional)

## Evaluation

```bash
# With Gazebo running:
python3 eval_vision_with_video.py \
    --model_path models/sac_vision/film_legacy_200k/actor.pth \
    --fusion_type film \
    --num_episodes 10
```

## Configuration

Edit `config.yaml` to change hyperparameters (learning rates, batch size, buffer size, network architecture, etc.).

## Troubleshooting

### Still not learning?
1. Check TensorBoard - is Q-value increasing?
2. Verify buffer is filling (check buffer_size in logs)
3. Try simpler fusion first (concat before film/attention)
4. Increase max_timesteps (vision needs more data)

### Out of memory?
1. Reduce batch_size (try 64)
2. Reduce buffer_size (try 50000)
3. Use resnet18 instead of resnet50

### Training too slow?
1. Use GPU (automatic if available)
2. Reduce update_every (but may be less stable)
3. Use simpler backbone (simple_cnn)

## Experiment Recommendations

1. **Baseline**: Start with concat fusion, resnet18, 100k timesteps
2. **Compare fusions**: Test film, attention, gated with same settings
3. **Longer training**: Run best fusion for 500k timesteps
4. **Architecture search**: Try resnet50 if resnet18 works well