FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV TURTLEBOT3_MODEL=burger

# Install dependencies including OSMesa for headless rendering
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    python3-rosdep \
    git \
    libosmesa6 \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglapi-mesa \
    gazebo11 \
    libgazebo11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages - use cu102 which works with system CUDA
RUN pip3 install --no-cache-dir \
    torch==1.6.0 torchvision==0.7.0 && \
    pip3 install --no-cache-dir \
    numpy==1.19.1 \
    scipy==1.5.2 \
    matplotlib==3.3.1 \
    pandas==1.1.1 \
    opencv-python-headless==4.4.0.42 \
    gym==0.17.2 \
    stable-baselines==2.10.1 \
    cloudpickle==1.3.0 \
    atari-py==0.2.6 \
    pillow==7.2.0 \
    pyglet==1.5.0 \
    joblib==0.16.0 \
    catkin_pkg \
    rospkg

# Create workspace and copy all project files
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws/src
COPY . /root/catkin_ws/src/

# Build workspace
WORKDIR /root/catkin_ws
RUN rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y || true && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Setup environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc && \
    echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc

WORKDIR /root/catkin_ws
CMD ["/bin/bash"]
