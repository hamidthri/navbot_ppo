#!/usr/bin/env python3
import os

import roslaunch
import rospy
import numpy as np
import math
from math import pi
import random
import time
# import tf.transformations
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel
from pick_laser import Pick
# from tf.transformations import euler_from_quaternion
from wall_penalty import pen_wall
diagonal_dis = math.sqrt(2) * (3.8 + 3.8)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
                              'turtlebot3_gazebo', 'models', 'Target', 'model.sdf')
len_batch = 6

class Env():
    def __init__(self, is_training, use_vision=False, vision_dim=64):
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.past_distance = 0.
        self.sum1 = 0
        self.sum2 = 0
        if is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4
        
        # Vision setup
        self.use_vision = use_vision
        self.vision_dim = vision_dim
        self.latest_image = None
        self.image_ok = False
        self.vision_encoder = None
        
        if self.use_vision:
            print(f"[Env] Initializing vision mode with feature dim={vision_dim}", flush=True)
            # Try to import and initialize vision encoder
            try:
                from vision_encoder import VisionEncoder
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.vision_encoder = VisionEncoder(output_dim=vision_dim, use_pretrained=True)
                self.vision_encoder = self.vision_encoder.to(device)
                self.vision_encoder.eval()
                self.vision_device = device
                print(f"[Env] Vision encoder loaded on {device}", flush=True)
            except Exception as e:
                print(f"[Env] ERROR loading vision encoder: {e}", flush=True)
                self.use_vision = False
            
            # Subscribe to camera topic
            # Real Gazebo camera topic: /robot_camera/image_raw (published by /gazebo plugin)
            self.image_topic = '/robot_camera/image_raw'
            self.sub_camera = rospy.Subscriber(self.image_topic, Image, self.imageCallback)
            print(f"[Env] Subscribed to {self.image_topic}", flush=True)
            
            # Wait briefly for first image
            print(f"[Env] Waiting for first image on {self.image_topic}...", flush=True)
            timeout = rospy.Time.now() + rospy.Duration(5.0)
            while self.latest_image is None and rospy.Time.now() < timeout and not rospy.is_shutdown():
                rospy.sleep(0.1)
            
            if self.latest_image is not None:
                self.image_ok = True
                print(f"[Env] First image received! encoding={self.latest_image.encoding}, "
                      f"size={self.latest_image.width}x{self.latest_image.height}, "
                      f"timestamp={self.latest_image.header.stamp.to_sec()}", flush=True)
            else:
                print(f"[Env] WARNING: No image received on {self.image_topic} within timeout", flush=True)
                print(f"[Env] Vision features will be zeros. Consider starting fake_camera_publisher.py", flush=True)

    def imageCallback(self, msg):
        """Store latest image from camera"""
        self.latest_image = msg
        if not self.image_ok:
            self.image_ok = True
            print(f"[Env] Image callback received: {msg.encoding}, {msg.width}x{msg.height}", flush=True)
    
    def getVisionFeatures(self):
        """Extract vision features from latest image, or return zeros if unavailable"""
        if not self.use_vision:
            return np.zeros(self.vision_dim)
        
        if self.latest_image is None or not self.image_ok:
            return np.zeros(self.vision_dim)
        
        try:
            features = self.vision_encoder.encode_image(self.latest_image, device=self.vision_device)
            return features
        except Exception as e:
            if not hasattr(self, '_vision_error_logged'):
                self._vision_error_logged = True
                print(f"[Env] Vision feature extraction error: {e}", flush=True)
            return np.zeros(self.vision_dim)

    # def close(self):
    #     """
    #     Close environment. No other method calls possible afterwards.
    #     """
    #     self.roslaunch.shutdown()
    #     time.sleep(10)

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)
        diff_angle = (yaw - rel_theta)
        if 0 <= diff_angle <= 180 or -180 <= diff_angle < 0:
            diff_angle = round(diff_angle, 2)
        elif diff_angle < -180:
            diff_angle = round(360 + diff_angle, 2)
        else:
            diff_angle = round(-360 + diff_angle, 2)

        # print(diff_angle)
        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    def getState(self, scan):
        scan_range = []
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.2
        done = False
        arrive = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        if current_distance <= self.threshold_arrive:
            arrive = True

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

    def setReward(self, done, arrive):
        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        distance_rate = (self.past_distance - current_distance)

        reward = 500.*distance_rate
        self.past_distance = current_distance

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        if arrive:
            reward = 120.
            self.pub_cmd_vel.publish(Twist())
            rospy.wait_for_service('/gazebo/delete_model')
            self.del_model('target')

            # Build the target
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'  # the same with sdf name
                target.model_xml = goal_urdf
                self.goal_position.position.x = random.uniform(-3.6, 3.6)
                self.goal_position.position.y = random.uniform(-3.6, 3.6)
                while 1.6 <= self.goal_position.position.x <= 2.4 and -1.4 <= self.goal_position.position.y <= 1.4 \
                        or -2.4 <= self.goal_position.position.x <= -1.6 and -1.4 <= self.goal_position.position.y <= 1.4 \
                        or -1.4 <= self.goal_position.position.x <= 1.4 and 1.6 <= self.goal_position.position.y <= 2.4 \
                        or -1.4 <= self.goal_position.position.x <= 1.4 and -2.4 <= self.goal_position.position.y <= -1.6:
                    self.goal_position.position.x = random.uniform(-3.6, 3.6)
                    self.goal_position.position.y = random.uniform(-3.6, 3.6)
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")
            rospy.wait_for_service('/gazebo/unpause_physics')
            self.goal_distance = self.getGoalDistace()
            arrive = False

        return reward

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]
        # state = Pick(state, len_batch)
        for pa in past_action:
            state.append(pa)
        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        
        # Append vision features if enabled
        if self.use_vision:
            vision_feats = self.getVisionFeatures()
            state = list(state) + list(vision_feats)
        
        reward = self.setReward(done, arrive)
        return np.asarray(state), reward, done, arrive

    def reset(self):
        # Reset the env #
        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('target')

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # Build the targets
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'target'  # the same with sdf name
            target.model_xml = goal_urdf

            self.goal_position.position.x = random.uniform(-3.6, 3.6)
            self.goal_position.position.y = random.uniform(-3.6, 3.6)
            while 1.7 <= self.goal_position.position.x <= 2.3 and -1.2 <= self.goal_position.position.y <= 1.2 \
                    or -2.3 <= self.goal_position.position.x <= -1.7 and -1.2 <= self.goal_position.position.y <= 1.2 \
                    or -1.2 <= self.goal_position.position.x <= 1.2 and 1.7 <= self.goal_position.position.y <= 2.3 \
                    or -1.2 <= self.goal_position.position.x <= 1.2 and -2.3 <= self.goal_position.position.y <= -1.7:
                self.goal_position.position.x = random.uniform(-3.6, 3.6)
                self.goal_position.position.y = random.uniform(-3.6, 3.6)
            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        rospy.wait_for_service('/gazebo/unpause_physics')
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]
        # state = Pick(state, len_batch)
        state.append(0)
        state.append(0)
        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        # Append vision features if enabled
        if self.use_vision:
            vision_feats = self.getVisionFeatures()
            state = list(state) + list(vision_feats)

        return np.asarray(state)
