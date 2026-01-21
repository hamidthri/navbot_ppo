# End-to-end Motion Planner Using Proximal Policy Optimization (PPO) in Gazebo

The goal is to use deep reinforcement learning algorithms, specifically Proximal Policy Optimization (PPO), to control a mobile robot (TurtleBot) to avoid obstacles while navigating towards a target.

**Goal:** Enable the robot (TurtleBot) to navigate to the target (enter the yellow circle).

---

## 🐳 **Running on Ubuntu 24.04? Use Docker!**

ROS1 cannot be installed natively on Ubuntu 24.04. We provide a complete **Docker setup with browser-accessible Gazebo GUI**.

**🚀 Quick Start (3 steps):**

```bash
# 1. Install Docker
./setup_docker.sh
# (Then log out and back in)

# 2. Build and run
./quick_start.sh

# 3. Open browser
# Go to: http://localhost:6080
# Run: roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch
# Then: roslaunch project ppo_stage_1.launch
```

**📖 See [QUICKSTART.md](QUICKSTART.md) for detailed Docker instructions.**

---


### Demo GIF

Watch a demonstration of our mapless motion planner in action (3x):

![Demo GIF](demo/video.gif)

### Introduction

## Advancements in Technology and Service Robots

As technology progresses, the integration of service robots into our daily lives is becoming increasingly common. Service robots incorporate a multitude of key technologies from various fields, including mobile navigation, system control, mechanism modules, vision modules, voice modules, artificial intelligence, and other related technical fields. This research is particularly focused on developing indoor robot navigation techniques.

## Project Overview: Learning-Based Mapless Motion Planner

In this project, we introduce a learning-based mapless motion planner that utilizes sparse laser signals and the target's position in the robot frame (i.e., relative distance and angles) as inputs. This approach generates continuous steering commands as outputs, eliminating the need for traditional mapping methods like SLAM (Simultaneous Localization and Mapping). This innovative planner is capable of navigating without prior maps and can adapt to new environments it has never encountered before.

Our approach is inspired by advancements in reinforcement learning for robotic navigation, as detailed in our paper ["Reinforcement Learning Based Mapless Navigation in Dynamic Environments Using LiDAR" (arXiv:2405.16266)](https://arxiv.org/abs/2405.16266). This paper demonstrates the efficacy of deep reinforcement learning techniques in enabling robots to navigate complex, dynamic environments without relying on pre-built maps.

### Input Specifications (State):

The input features, forming a 16-dimensional state, include:

1. **Laser Finding (10 Dimensions)** - Represents sparse laser measurements.
2. **Past Action (2 Dimensions)**:
   - Linear velocity
   - Angular velocity
3. **Target Position in Robot Frame (2 Dimensions)**:
   - Relative distance
   - Relative angle (using polar coordinates)
4. **Robot Yaw Angular (1 Dimension)** - Indicates the robot's current yaw angle.
5. **Degrees to Face the Target (1 Dimension)** - The absolute difference between the yaw and the relative angle to the target.

### Normalization of Inputs:

Normalization is applied to the inputs to facilitate learning by scaling all values to a consistent range:

1. **Laser Finding** - Divided by the maximum laser finding range.
2. **Past Action** - Retained as original values.
3. **Target Position in Robot Frame**:
   - Relative distance normalized by the diagonal length of the map.
   - Relative angle normalized by 360 degrees.
4. **Robot Yaw Angular** - Normalized by 360 degrees.
5. **Degrees to Face the Target** - Normalized by 180 degrees.

### Output Specifications (Action):

The outputs, forming a 2-dimensional action, consist of:

1. **Linear Velocity (1 Dimension)** - Ranging from 0 to 0.25 meters per second.
2. **Angular Velocity (1 Dimension)** - Ranging from -0.5 to 0.5 radians per second.


## Reward System
- **Arrive at the target**: +120
- **Hit the wall**: -100
- **Move towards target**: 500 * (Past relative distance - Current relative distance)

## Algorithm
- **Proximal Policy Optimization (PPO)** using Actor and Critic methods, implemented with PyTorch.

## Training Environment
- **Gazebo**: A robot simulation environment, managed via Neotic and PyTorch for enhanced machine learning capabilities.

### Installation Dependencies:

1. **Python3**
2. **PyTorch**:
   ```bash
   pip3 install torch torchvision

3) ROS noetic

> http://wiki.ros.org/noetic/Installation/Ubuntu

4) Gazebo7 (When you install ros kinetic it also install gazebo7)

> http://gazebosim.org/tutorials?cat=install&tut=install_ubuntu&ver=7.0


# Installation

## Option 1: Docker Setup (Recommended for Ubuntu 24.04)

**For Ubuntu 24.04 users:** Since ROS1 cannot be installed natively on Ubuntu 24, use the Docker setup with browser-based Gazebo GUI.

📖 **See [DOCKER_SETUP.md](DOCKER_SETUP.md) for complete Docker installation and usage instructions.**

**Quick Start:**
```bash
# Build and start container
docker compose up -d

# Access container
docker exec -it navbot-ppo bash

# Open browser to http://localhost:6080 to view Gazebo GUI

# Inside container - Terminal 1:
roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch

# Inside container - Terminal 2:
roslaunch project ppo_stage_1.launch
```

## Option 2: Native Installation (Ubuntu 20.04 only)

# Create and Initialize the Workspace
```bash
cd
mkdir -p catkin_ws/src && cd catkin_ws/src
git clone https://github.com/hamidthri/navbot_ppo.git project
git clone https://github.com/hamidthri/turtlebot3
git clone https://github.com/hamidthri/turtlebot3_msgs
git clone https://github.com/hamidthri/turtlebot3_simulations
cd ..
catkin_make
```

# Configure Environment
```bash
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
echo "source /home/'Enter your user name'/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 🐋 Docker Development Setup

For development where you want to see code changes immediately on your host machine, use a bind-mounted container:

### One-Time Host Setup (Linux)

```bash
# Allow X11 forwarding for Gazebo GUI
xhost +local:root

# To revert later (after stopping container):
# xhost -local:root
```

### Running Development Container with Bind Mount

```bash
# From your repo root
docker run -it --rm \
  --name navbot-ppo-dev \
  --net=host \
  -e DISPLAY=:1 \
  -e GAZEBO_PLUGIN_PATH=/opt/ros/noetic/lib:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins \
  -e LD_LIBRARY_PATH=/opt/ros/noetic/lib:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:${LD_LIBRARY_PATH} \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/root/catkin_ws/src:rw \
  navbot-ppo:latest \
  bash
```

**Important Notes:**
- `DISPLAY=:1` - Adjust to match your X server (check `/tmp/.X11-unix/`)
- Bind mount creates live sync between host and container
- Changes in your IDE on host appear immediately in container
- `GAZEBO_PLUGIN_PATH` and `LD_LIBRARY_PATH` are **required** for camera plugins

### Inside Container

```bash
# Source ROS environment
source /opt/ros/noetic/setup.bash
cd /root/catkin_ws
catkin_make
source devel/setup.bash

# Launch training world with robot camera
roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch

# In another terminal (docker exec -it navbot-ppo-dev bash)
cd /root/catkin_ws/src/project_ppo/src
python3 main.py --method_name my_experiment
```

---

## 📷 Gazebo Camera Troubleshooting

The robot uses a **real Gazebo camera plugin** for vision-based training. If you encounter issues:

### Symptoms
- No `/robot_camera/image_raw` topic
- Error: "Unable to create CameraSensor. Rendering is disabled"
- Plugin not loading: "libCameraPlugin.so: cannot open shared object file"

### Diagnostic Steps

**1. Check Environment Variables**
```bash
# Inside container
echo $DISPLAY                    # Should be :1 (or match your X server)
echo $GAZEBO_PLUGIN_PATH        # Should include /opt/ros/noetic/lib and gazebo-11 plugins
echo $LD_LIBRARY_PATH           # Should include plugin paths
```

**2. Verify Plugin Dependencies**
```bash
# Check if camera plugin can find all dependencies
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:$LD_LIBRARY_PATH
ldd /opt/ros/noetic/lib/libgazebo_ros_camera.so | grep "not found"

# Should show nothing. If you see "not found", the paths are incorrect.
```

**3. Check Camera Topic**
```bash
source /opt/ros/noetic/setup.bash

# List camera topics (should see /robot_camera/image_raw)
rostopic list | grep camera

# Verify publisher is Gazebo (NOT fake_camera_publisher)
rostopic info /robot_camera/image_raw
# Output should show: Publishers: * /gazebo (http://...)

# Check publishing rate (~8-10 Hz)
rostopic hz /robot_camera/image_raw
```

**4. Check Gazebo Logs**
```bash
# Look for camera/rendering errors
find ~/.ros/log -name "gazebo*.log" -mmin -5 | xargs grep -i "camera\|render\|plugin"
```

### Required Environment Variables (Fix)

If camera is not working, ensure these are set **before launching Gazebo**:

```bash
export DISPLAY=:1  # Match your X server socket in /tmp/.X11-unix/
export GAZEBO_PLUGIN_PATH=/opt/ros/noetic/lib:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:$LD_LIBRARY_PATH
```

### Launch File Requirements

In your Gazebo launch file (e.g., `turtlebot3_stage_1.launch`), ensure:
```xml
<arg name="headless" value="false"/>  <!-- NOT true! Camera needs rendering -->
<arg name="gui" value="false"/>       <!-- Can be false, but headless must be false -->
```

### Save and Verify a Camera Frame

```bash
# Inside container with ROS running
cd /root/catkin_ws/src/project_ppo/src
python3 << 'EOF'
import rospy
from sensor_msgs.msg import Image
import numpy as np
from PIL import Image as PILImage

rospy.init_node('save_frame', anonymous=True)
frame = None

def cb(msg):
    global frame
    if frame is None:
        frame = msg
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        PILImage.fromarray(arr, 'RGB').save('/tmp/camera_test.png')
        print(f'Saved {msg.width}x{msg.height} frame')
        rospy.signal_shutdown('done')

rospy.Subscriber('/robot_camera/image_raw', Image, cb)
rospy.spin()
EOF

# Copy to host
exit  # exit container
docker cp navbot-ppo-dev:/tmp/camera_test.png ./camera_test.png
```

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| No camera topic | GAZEBO_PLUGIN_PATH not set | Export path before roslaunch |
| "Rendering is disabled" | headless=true or DISPLAY wrong | Set headless=false, check DISPLAY |
| libCameraPlugin.so not found | LD_LIBRARY_PATH missing gazebo plugins | Add /usr/lib/x86_64-linux-gnu/gazebo-11/plugins |
| Topic exists but no messages | Camera sensor not attached to robot | Check URDF has camera_link + plugin |

---

## Running the Demo
```bash
roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch
```
## In another terminal
```bash
roslaunch project_ppo ppo_stage_1.launch
```

# Training Option
If you want to retrain the model, modify the following in the specified file:
```bash
is_training = True
```
Located in: project/src/ppo_stage_1.py

_______________________________________________________

### Reference:

#### Idea:
- [PPO Paper](https://arxiv.org/pdf/1703.00420.pdf)

#### ROS Workspace:
- [Turtlebot3 GitHub Repository](https://github.com/ROBOTIS-GIT/turtlebot3)
- [Turtlebot3 Messages Repository](https://github.com/ROBOTIS-GIT/turtlebot3_msgs)
- [Turtlebot3 Simulations Repository](https://github.com/ROBOTIS-GIT/turtlebot3_simulations)
- [Project GitHub Repository](https://github.com/dranaju/project)


This version keeps all the command line instructions within a single code block for simplicity and adds the installation command for PyTorch. Adjust the paths or any commands as necessary to accurately reflect your project setup and environment.

