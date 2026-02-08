# Quick Start Guide

This guide will help you set up and run the SAC navigation implementations.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU (optional, for faster training)
- Linux host (tested on Ubuntu)

## Step 1: Clone the Repository

```bash
git clone https://github.com/hamidthri/navbot_ppo.git
cd navbot_ppo
```

## Step 2: Build the Docker Image

```bash
docker compose build
```

This will take ~10-15 minutes as it installs ROS, Gazebo, PyTorch, and all dependencies.

## Step 3: Start the Container

```bash
docker compose up -d
```

## Step 4: Verify Setup

Enter the container and check if all directories are mounted:

```bash
docker exec -it navbot-ppo bash
ls /root/catkin_ws/src/
# Should see: sac_vision, sac_lidar, project, turtlebot3, etc.
```

## Step 5a: Run SAC Vision Training

```bash
# Inside the container
cd /root/catkin_ws/src/sac_vision
./run_vision_training.sh --fusion_type film --max_timesteps 200000 --reward_type legacy
```

Or directly from host:

```bash
docker exec -it navbot-ppo bash -c "cd /root/catkin_ws/src/sac_vision && ./run_vision_training.sh --max_timesteps 200000"
```

## Step 5b: Run SAC LiDAR Training

```bash
# Inside the container
cd /root/catkin_ws/src/sac_lidar
./run_lidar_training.sh --max_timesteps 200000 --reward_type legacy
```

## Monitoring Training

### TensorBoard

```bash
# In another terminal
docker exec -it navbot-ppo bash
tensorboard --logdir /root/catkin_ws/src/sac_vision/models/sac_vision
# Visit http://localhost:6006
```

### Training Logs

Logs are saved in:
- Vision: `sac_vision/models/sac_vision/{run_name}/training_*.log`
- LiDAR: `sac_lidar/models/sac_lidar/training_*.log`

## Common Issues

### GPU Not Available

If you see NVIDIA driver errors, edit `docker-compose.yml` and uncomment the GPU sections only if your drivers are working.

### Permission Issues

Files created inside the container are owned by the container user. To change ownership:

```bash
sudo chown -R $USER:$USER sac_vision/models sac_lidar/models
```

### Container Already Running

```bash
docker stop navbot-ppo
docker rm navbot-ppo
docker compose up -d
```

## Next Steps

- See [sac_vision/README.md](sac_vision/README.md) for detailed vision training options
- See [sac_lidar/README.md](sac_lidar/README.md) for detailed lidar training options
- Check training results in the `models/` directories
- Use evaluation scripts to test trained models
