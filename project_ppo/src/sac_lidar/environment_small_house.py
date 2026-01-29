#!/usr/bin/env python3
"""
Small House Environment for Navigation Training
Based on environment_new.py with adaptations for AWS RoboMaker Small House world
"""
import os
import rospy
import numpy as np
import math
from math import pi
import random
import time
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState
from gazebo_msgs.msg import ModelState
from small_house_region_sampler import SmallHouseRegionSampler

# Diagonal distance for small house (approximately 17m x 10m)
diagonal_dis = math.sqrt(2) * (17.0 + 10.0)

# Use absolute path to goal model (works from any folder location)
goal_model_dir = '/root/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/Target/model.sdf'


class Env():
    def __init__(self, is_training):
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
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.past_distance = 0.
        
        if is_training:
            self.threshold_arrive = 0.3  # Slightly larger for small house
        else:
            self.threshold_arrive = 0.5
        
        # Initialize region sampler for small house
        self.region_sampler = SmallHouseRegionSampler(
            initial_distance=2.0,
            max_distance=18.0,
            distance_increment=0.1,
            increment_every_n_episodes=500
        )
        
        print(f"[SmallHouseEnv] Initialized with arrival threshold: {self.threshold_arrive}m")

    def getGoalDistace(self):
        goal_distance = math.hypot(
            self.goal_position.position.x - self.position.x, 
            self.goal_position.position.y - self.position.y
        )
        self.past_distance = goal_distance
        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 
                                            1 - 2 * (q_y * q_y + q_z * q_z))))

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

        current_distance = math.hypot(
            self.goal_position.position.x - self.position.x, 
            self.goal_position.position.y - self.position.y
        )
        if current_distance <= self.threshold_arrive:
            arrive = True

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

    def setReward(self, done, arrive):
        current_distance = math.hypot(
            self.goal_position.position.x - self.position.x, 
            self.goal_position.position.y - self.position.y
        )
        distance_rate = (self.past_distance - current_distance)

        reward = 500. * distance_rate
        self.past_distance = current_distance

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        if arrive:
            reward = 120.
            self.pub_cmd_vel.publish(Twist())
            
            # Update curriculum learning
            self.region_sampler.update_curriculum(success=True)
            
            # Delete old goal
            rospy.wait_for_service('/gazebo/delete_model')
            self.del_model('target')

            # Spawn new goal using region sampler
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'
                target.model_xml = goal_urdf
                
                # Use region sampler to get goal position based on curriculum
                goal_x, goal_y, region_name = self.region_sampler.get_goal_position(
                    self.position.x, 
                    self.position.y
                )
                
                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y
                self.goal_position.position.z = 0.0
                
                self.goal(target.model_name, target.model_xml, 'namespace', 
                         self.goal_position, 'world')
                
                # print(f"[SmallHouseEnv] New goal at ({goal_x:.2f}, {goal_y:.2f}) in {region_name}")
                
            except (rospy.ServiceException) as e:
                print(f"/gazebo/failed to build the target: {e}")
                
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

        scan_range, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in scan_range]

        for pa in past_action:
            state.append(pa)

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        reward = self.setReward(done, arrive)

        return np.asarray(state), reward, done, arrive

    def reset(self):
        # Reset simulation
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # Get random spawn position for robot
        robot_x, robot_y, spawn_name = self.region_sampler.get_robot_spawn_position()
        
        # Move robot to spawn position
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            state_msg = ModelState()
            state_msg.model_name = 'turtlebot3_burger'
            state_msg.pose.position.x = robot_x
            state_msg.pose.position.y = robot_y
            state_msg.pose.position.z = 0.0
            state_msg.pose.orientation.x = 0.0
            state_msg.pose.orientation.y = 0.0
            state_msg.pose.orientation.z = 0.0
            state_msg.pose.orientation.w = 1.0
            self.set_state(state_msg)
            # print(f"[SmallHouseEnv] Reset: Robot moved to {spawn_name} at ({robot_x:.2f}, {robot_y:.2f})")
        except (rospy.ServiceException) as e:
            print(f"Failed to set robot position: {e}")

        # Delete old goal if exists
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            self.del_model('target')
        except:
            pass

        # Spawn new goal using region sampler
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'target'
            target.model_xml = goal_urdf
            
            # Sample goal based on robot spawn position
            goal_x, goal_y, region_name = self.region_sampler.get_goal_position(
                robot_x, 
                robot_y
            )
            
            self.goal_position.position.x = goal_x
            self.goal_position.position.y = goal_y
            self.goal_position.position.z = 0.0
            
            self.goal(target.model_name, target.model_xml, 'namespace', 
                     self.goal_position, 'world')
            
            distance = math.hypot(goal_x - robot_x, goal_y - robot_y)
            # print(f"[SmallHouseEnv] Goal at ({goal_x:.2f}, {goal_y:.2f}) in {region_name}, distance: {distance:.2f}m")
            
        except (rospy.ServiceException) as e:
            print(f"/gazebo/failed to build the target: {e}")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/unpause_physics service call failed")

        # Wait for scan data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        self.goal_distance = self.getGoalDistace()
        scan_range, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in scan_range]
        state.append(0)
        state.append(0)
        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)
