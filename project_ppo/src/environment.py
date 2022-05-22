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
# from tf.transformations import euler_from_quaternion
from wall_penalty import pen_wall
diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
                              'turtlebot3_gazebo', 'models', 'Target', 'model.sdf')


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
        self.past_distance = 0.
        self.sum1 = 0
        self.sum2 = 0
        if is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4

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
        # roll, pich, yaw = tf.transformations.euler_from_quaternion(q_x, q_y, q_z, q_w)
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
        # diff_angle = abs(rel_theta - yaw)
        # if diff_angle <= 180:
        #     diff_angle = round(diff_angle, 2)
        # else:
        #     diff_angle = round(-360 + diff_angle, 2)
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

    def setReward(self, done, arrive, past_state, new_state, t_so_far):
        current_distance = math.hypot(self.goal_position.position.x - self.position.x,
                                      self.goal_position.position.y - self.position.y)
        # past_pen_dis = pen_wall(past_state)
        current_pen_dis, value_middle = pen_wall(new_state)

        # wall_rate_pen = past_pen_dis - current_pen_dis

        # norm_dist = 1
        # threshold = 2.
        # if current_distance < threshold:
        #     norm_dist = current_distance/threshold
        # elif current_distance < .2:
        #     norm_dist = .2/threshold

        # wall_rate_pen = (past_pen_dis - current_pen_dis)
        wall_rate_pen = - current_pen_dis * (1 - value_middle)
        # self.sum1 = self.sum1 + wall_rate_pen
        if (self.past_distance - current_distance) >= 0:
            distance_rate = (self.past_distance - current_distance) * (4 * math.sqrt(2) - current_distance)
        else:
            distance_rate = (self.past_distance - current_distance) * current_distance
        # self.sum2 = self.sum2 + distance_rate
        time_step_pen = 1

        # if -20 <= self.diff_angle <= 20:
        #     reward = 500. * distance_rate - time_step_pen
        # else:
        reward = 200.*distance_rate + 2. * wall_rate_pen - time_step_pen
        # reward = 500 * wall_rate_pen
        self.past_distance = current_distance

        if done:
            reward = -500.
            self.pub_cmd_vel.publish(Twist())

        if arrive:
            reward = 500.
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
                # if t_so_far <= 100000:
                #     goal_space = [[2, 2], [0, 2.3], [1.7, 0], [1.7, 1.3], [2.3, 1.3], [2, -2], [0, -2.3], [0, 3.6],
                #                   [-1.7, -1.3], [-2.3, 0], [-1.7, 1.3], [-3.6, 3.6]]
                #     goal_pos = goal_space[np.random.choice(len(goal_space))]
                #     self.goal_position.position.x = goal_pos[0]
                #     self.goal_position.position.y = goal_pos[1]
                #     self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
                # else:
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
            self.goal_distance = self.getGoalDistace()
            arrive = False

        return reward

    def step(self, action, past_action, old_state, t_so_far):
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
        for pa in past_action:
            state.append(pa)
        new_state = state[: -2]
        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        reward = self.setReward(done, arrive, old_state, new_state, t_so_far)
        return np.asarray(state), reward, done, arrive, new_state

    def reset(self, t_so_far):
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
            # if t_so_far <= 100000:
            #     goal_space = [[2, 2], [0, 2.3], [1.7, 0], [1.7, 1.3], [2.3, 1.3], [2, -2], [0, -2.3], [0, 3.6],
            #                   [-1.7, -1.3], [-2.3, 0], [-1.7, 1.3], [-3.6, 3.6]]
            #     goal_pos = goal_space[np.random.choice(len(goal_space))]
            #     self.goal_position.position.x = goal_pos[0]
            #     self.goal_position.position.y = goal_pos[1]
            #     self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            # else:
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
        state.append(0)
        state.append(0)
        past_state = state[: -2]

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state), past_state
