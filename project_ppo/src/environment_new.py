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
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState
from pick_laser import Pick
# from tf.transformations import euler_from_quaternion
from wall_penalty import pen_wall
diagonal_dis = math.sqrt(2) * (3.8 + 3.8)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
                              'turtlebot3_gazebo', 'models', 'Target', 'model.sdf')
len_batch = 36  # 360 laser points / 36 = 10 picked laser features

class Env():
    def __init__(self, is_training, use_vision=False, vision_dim=64, sampler_mode='legacy', debug_sampler=False, distance_uniform=False, reward_type='legacy', fixed_case_path=None, method_run_dir=None,
                 curriculum_min_dist=0.5, curriculum_max_dist=5.0, curriculum_steps=100000):
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.past_distance = 0.
        self.sum1 = 0
        self.sum2 = 0
        if is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4
        
        # Simple distance curriculum: gradually increase goal distance
        self.curriculum_min_dist = curriculum_min_dist
        self.curriculum_max_dist = curriculum_max_dist
        self.curriculum_steps = curriculum_steps
        self.training_step = 0
        print(f"[Env] Distance curriculum: {curriculum_min_dist:.1f}m -> {curriculum_max_dist:.1f}m over {curriculum_steps} steps", flush=True)
        
        # Reward system setup
        self.reward_type = reward_type
        if reward_type == 'fuzzy3':
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rewards'))
            from fuzzy3_reward import Fuzzy3Reward
            self.reward_fn = Fuzzy3Reward()
            print(f"[Env] Reward: Fuzzy3Reward (original)", flush=True)
        elif reward_type == 'fuzzy3_v4':
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rewards'))
            from fuzzy3_reward_v4 import Fuzzy3RewardV4
            self.reward_fn = Fuzzy3RewardV4(theta_is_relative=True)
            print(f"[Env] Reward: Fuzzy3RewardV4 (anti-stall, robust clearance, conditional danger)", flush=True)
        else:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rewards'))
            from legacy_reward import LegacyReward
            self.reward_fn = LegacyReward()
            print(f"[Env] Reward: LegacyReward", flush=True)
        
        # Sampler setup
        self.sampler_mode = sampler_mode
        self.debug_sampler = debug_sampler
        self.distance_uniform = distance_uniform
        self.map_sampler = None
        self.rect_sampler = None
        self.gazebo_helpers = None
        
        # Legacy compatibility: map 'legacy' sampler_mode to random sampling
        self.use_map_sampler = (sampler_mode == 'map')
        
        if sampler_mode == 'map':
            from map_goal_sampler import MapGoalSampler
            import gazebo_reset_helpers
            import os
            
            # Load map
            src_dir = os.path.dirname(os.path.abspath(__file__))
            map_yaml = os.path.join(src_dir, '..', 'maps', 'small_house.yaml')
            
            self.map_sampler = MapGoalSampler(
                map_yaml_path=map_yaml,
                clearance_start=0.50,  # Minimum 0.5m clearance for robot start
                clearance_goal=0.30,   # Safe clearance above collision threshold (0.2m)
                min_distance=2.5,
                max_distance=8.0,
                distance_uniform=distance_uniform,
                distance_bins=10,
                min_bin_candidates=10
            )
            self.gazebo_helpers = gazebo_reset_helpers
            mode_str = "distance-uniform" if distance_uniform else "spatial-uniform"
            print(f"[Env] Map-based sampler enabled ({mode_str}, clearance: start=0.50m, goal=0.30m, dist=2.5-8.0m)", flush=True)
        
        elif sampler_mode == 'rect_regions':
            from rect_region_sampler import RectRegionSampler
            import gazebo_reset_helpers
            
            self.rect_sampler = RectRegionSampler(
                robot_radius=0.105,
                clearance_margin=0.30,
                max_tries=100
            )
            self.gazebo_helpers = gazebo_reset_helpers
            print(f"[Env] Rectangle region sampler enabled (13 regions, start != goal)", flush=True)
        
        # Fixed case capture/replay setup
        self.fixed_case_path = fixed_case_path
        self.method_run_dir = method_run_dir
        self.fixed_case = None
        self.fixed_case_saved = False
        self.episode_count = 0
        
        if fixed_case_path:
            # Replay mode: load fixed case
            import json
            try:
                with open(fixed_case_path, 'r') as f:
                    self.fixed_case = json.load(f)
                print(f"[Env] Fixed case replay mode enabled: {fixed_case_path}", flush=True)
                print(f"[Env]   Start: ({self.fixed_case['start_x']:.2f}, {self.fixed_case['start_y']:.2f}, {self.fixed_case['start_yaw']:.1f}°)", flush=True)
                print(f"[Env]   Goal: ({self.fixed_case['goal_x']:.2f}, {self.fixed_case['goal_y']:.2f})", flush=True)
            except Exception as e:
                print(f"[Env] ERROR: Failed to load fixed case from {fixed_case_path}: {e}", flush=True)
                self.fixed_case = None
        
        # Vision setup
        self.use_vision = use_vision
        self.vision_dim = vision_dim
        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_image_encoding = None  # Track image encoding from ROS message
        self.latest_image_shape = None  # Track image shape (H, W, C)
        self.image_ok = False
        
        if self.use_vision:
            print(f"[Env] Initializing vision mode (raw image only, no encoder in env)", flush=True)
            # Subscribe to camera topic
            # TurtleBot3 burger camera topic: /robot_camera/image_raw (published by Gazebo libgazebo_ros_camera.so)
            self.image_topic = '/robot_camera/image_raw'
            self.sub_camera = rospy.Subscriber(self.image_topic, Image, self.imageCallback)
            print(f"[Env] Subscribed to {self.image_topic}", flush=True)
            
            # Note: We don't use cv_bridge due to boost initialization issues
            # Instead, we manually convert ROS Image messages to numpy arrays
            
            # STRICT: Wait for first image (10s timeout) - training MUST fail if no camera
            print(f"[Env] Waiting for first image on {self.image_topic} (timeout: 10s)...", flush=True)
            timeout = rospy.Time.now() + rospy.Duration(10.0)
            while self.latest_image is None and rospy.Time.now() < timeout and not rospy.is_shutdown():
                rospy.sleep(0.1)
            
            if self.latest_image is not None:
                self.image_ok = True
                print(f"[Env] ✓ First camera frame received!", flush=True)
                print(f"[Env]   - encoding: {self.latest_image_encoding}", flush=True)
                print(f"[Env]   - size: {self.latest_image_shape[1]}x{self.latest_image_shape[0]}", flush=True)
                print(f"[Env]   - timestamp: {self.latest_image_stamp:.2f}", flush=True)
            else:
                # HARD FAILURE: No camera means training cannot proceed
                raise RuntimeError(
                    f"\n{'='*70}\n"
                    f"CAMERA ERROR: No frames received on {self.image_topic} within 10s timeout.\n"
                    f"Vision training requires real camera images from Gazebo.\n"
                    f"\n"
                    f"Diagnostics:\n"
                    f"  - Check if Gazebo is running with GUI: roslaunch project_ppo navbot_small_house.launch gui:=true\n"
                    f"  - Verify camera topic exists: rostopic list | grep camera\n"
                    f"  - Check topic frequency: rostopic hz {self.image_topic}\n"
                    f"  - Ensure robot URDF includes camera sensor with libgazebo_ros_camera.so plugin\n"
                    f"\n"
                    f"Training ABORTED: Cannot proceed without real camera frames.\n"
                    f"{'='*70}"
                )

    def imageCallback(self, msg):
        """Store latest image from camera as numpy array (manual conversion without cv_bridge)"""
        try:
            # Manual conversion from ROS Image to numpy array (avoids cv_bridge boost issues)
            # ROS Image message structure:
            #   - msg.data: flat bytes array
            #   - msg.encoding: pixel format (e.g., 'rgb8', 'bgr8')
            #   - msg.height, msg.width: image dimensions
            #   - msg.step: row stride in bytes
            
            import numpy as np
            
            # Convert bytes to numpy array
            if msg.encoding == 'rgb8':
                # Already RGB, just reshape
                image_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                self.latest_image = image_array.copy()  # Copy to avoid reference issues
            elif msg.encoding == 'bgr8':
                # Need to convert BGR to RGB
                image_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                # Swap R and B channels: BGR -> RGB
                self.latest_image = image_array[:, :, ::-1].copy()  # Reverse last dimension
            else:
                # Unsupported encoding
                print(f"[Env] WARNING: Unsupported image encoding: {msg.encoding}", flush=True)
                return
            
            # Store metadata
            self.latest_image_stamp = msg.header.stamp.to_sec()
            self.latest_image_encoding = msg.encoding
            self.latest_image_shape = self.latest_image.shape
            
            if not self.image_ok:
                self.image_ok = True
                print(f"[Env] ✓ Image callback received: {msg.encoding}, {msg.width}x{msg.height}, converted to RGB numpy array (shape: {self.latest_image.shape})", flush=True)
                
        except Exception as e:
            if not hasattr(self, '_img_convert_error_logged'):
                self._img_convert_error_logged = True
                print(f"[Env] Image conversion error: {e}", flush=True)
    
    def getLatestImage(self):
        """Return latest image as numpy uint8 HxWx3 RGB, or None if unavailable"""
        if not self.use_vision or self.latest_image is None:
            return None
        return self.latest_image
    
    def update_curriculum_step(self, step):
        """Update training step for curriculum (called from PPO)"""
        self.training_step = step
    
    def get_current_max_distance(self):
        """Get current maximum goal distance based on curriculum progress"""
        progress = min(1.0, self.training_step / self.curriculum_steps)
        current_max = self.curriculum_min_dist + progress * (self.curriculum_max_dist - self.curriculum_min_dist)
        return current_max
    
    def getImageAge(self):
        """Return age of latest image in seconds, or None if unavailable"""
        if not hasattr(self, 'latest_image_stamp'):
            return None
        import rospy
        current_time = rospy.Time.now().to_sec()
        return current_time - self.latest_image_stamp
    
    def getVisionFeatures(self):
        """Deprecated - vision features now computed in policy network"""
        # Return zeros for backward compatibility
        if not self.use_vision:
            return np.zeros(self.vision_dim)
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
    
    def verify_target_pose(self, label=""):
        """Verify target model exists and get its pose"""
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=1.0)
            resp = self.get_model_state('target', 'world')
            if resp.success:
                pos = resp.pose.position
                print(f"[Env] verify_target_pose [{label}]: target at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})", flush=True)
                return True, pos
            else:
                print(f"[Env] verify_target_pose [{label}]: get_model_state failed - {resp.status_message}", flush=True)
                return False, None
        except Exception as e:
            print(f"[Env] verify_target_pose [{label}]: Exception - {e}", flush=True)
            return False, None

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

    def setReward(self, done, arrive, scan_range=None):
        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        
        # Use reward function based on type
        if self.reward_type in ['fuzzy3', 'fuzzy3_v4']:
            # Fuzzy reward needs: current_distance, yaw, rel_theta, scan_range, done, arrive
            reward = self.reward_fn.compute(
                current_distance=current_distance,
                current_yaw=self.yaw,
                current_rel_theta=self.rel_theta,
                scan_range=scan_range,
                done=done,
                arrive=arrive
            )
        else:
            # Legacy reward (original logic)
            reward = self.reward_fn.compute(
                current_distance=current_distance,
                done=done,
                arrive=arrive
            )
        
        # Legacy behavior for arrive: respawn goal
        if arrive:
            self.pub_cmd_vel.publish(Twist())
            
            # Pause physics before manipulating models
            try:
                rospy.wait_for_service('/gazebo/pause_physics', timeout=1.0)
                self.pause_proxy()
            except:
                pass
            
            # Delete old target
            rospy.wait_for_service('/gazebo/delete_model')
            try:
                self.del_model('target')
            except:
                pass

            # Build the new target
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'
                target.model_xml = goal_urdf
                self.goal_position.position.x = random.uniform(-3.6, 3.6)
                self.goal_position.position.y = random.uniform(-3.6, 3.6)
                self.goal_position.position.z = 0.01
                while 1.6 <= self.goal_position.position.x <= 2.4 and -1.4 <= self.goal_position.position.y <= 1.4 \
                        or -2.4 <= self.goal_position.position.x <= -1.6 and -1.4 <= self.goal_position.position.y <= 1.4 \
                        or -1.4 <= self.goal_position.position.x <= 1.4 and 1.6 <= self.goal_position.position.y <= 2.4 \
                        or -1.4 <= self.goal_position.position.x <= 1.4 and -2.4 <= self.goal_position.position.y <= -1.6:
                    self.goal_position.position.x = random.uniform(-3.6, 3.6)
                    self.goal_position.position.y = random.uniform(-3.6, 3.6)
                
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
                
            except (rospy.ServiceException) as e:
                pass
            
            # Unpause physics after models are set up
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                self.unpause_proxy()
            except:
                pass
            
            self.goal_distance = self.getGoalDistace()
            
            # Reset reward function state for new goal
            self.reward_fn.reset(self.goal_distance)
            
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

        raw_scan_range, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        
        # Normalize scan for policy (NOT for reward function)
        state = [i / 3.5 for i in raw_scan_range]
        
        # Uniform sampling: select 10 evenly-spaced samples from normalized scan
        L = len(state)
        indices = [int(i * L / 10) for i in range(10)]
        lidar_features = [state[idx] for idx in indices]
        
        # Build base state: 10 lidar + 2 past_action + 4 goalpose
        assert len(lidar_features) == 10, f"LiDAR features must be 10, got {len(lidar_features)}"
        base_state = lidar_features.copy()
        for pa in past_action:
            base_state.append(pa)
        base_state = base_state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        
        # Assertion: Verify base state is exactly 16-d (10 lidar + 2 past_action + 4 goalpose)
        assert len(base_state) == 16, f"Base state must be 16-d, got {len(base_state)}"
        
        # Vision mode: return base state vector only (NO vision features appended here)
        # Image will be handled separately in PPO rollout
        
        # CRITICAL: Pass raw scan_range (meters) to reward, not normalized state
        reward = self.setReward(done, arrive, scan_range=raw_scan_range)
        return np.asarray(base_state), reward, done, arrive

    def reset(self):
        # Reset the env
        
        # Increment episode counter
        self.episode_count += 1
        
        # Delete old target if it exists
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            self.del_model('target')
        except:
            pass
        
        # Fixed case replay mode
        if self.fixed_case:
            start_x = self.fixed_case['start_x']
            start_y = self.fixed_case['start_y']
            start_yaw = self.fixed_case['start_yaw']
            goal_x = self.fixed_case['goal_x']
            goal_y = self.fixed_case['goal_y']
            
            if self.episode_count == 1:
                print(f"[RESET] Replaying fixed case: start=({start_x:.2f},{start_y:.2f},{start_yaw:.1f}°) goal=({goal_x:.2f},{goal_y:.2f})", flush=True)
        
        # Map-based sampler: sample start/goal with clearance guarantees
        elif self.use_map_sampler and self.map_sampler is not None:
            import math
            
            # Retry logic: try up to 3 times to get valid reset (no immediate collision)
            max_reset_attempts = 3
            for attempt in range(max_reset_attempts):
                # Sample start pose
                start_pose, start_tries = self.map_sampler.sample_start()
                start_x, start_y, start_yaw = start_pose
                
                # Sample goal position (distance-uniform or spatial-uniform)
                if self.distance_uniform:
                    if self.debug_sampler:
                        goal_xy, goal_tries, _, reach_info = self.map_sampler.sample_goal_distance_uniform(start_pose[:2], debug=True)
                    else:
                        goal_xy, goal_tries, _ = self.map_sampler.sample_goal_distance_uniform(start_pose[:2], debug=False)
                        reach_info = None
                else:
                    if self.debug_sampler:
                        goal_xy, goal_tries, _, reach_info = self.map_sampler.sample_goal(start_pose[:2], debug=True)
                    else:
                        goal_xy, goal_tries, _ = self.map_sampler.sample_goal(start_pose[:2], debug=False)
                        reach_info = None
                
                goal_x, goal_y = goal_xy
                tries = start_tries + goal_tries
                
                # Optional debug logging
                if self.debug_sampler:
                    dist = math.hypot(goal_x - start_x, goal_y - start_y)
                    clearance_start = self.map_sampler.get_clearance(start_x, start_y)
                    clearance_goal = self.map_sampler.get_clearance(goal_x, goal_y)
                    
                    reach_str = ""
                    if reach_info:
                        if self.distance_uniform:
                            # Distance-uniform mode: show bin info
                            reach_str = f" dist_uniform=(bin:{reach_info.get('chosen_bin', -1)}, cands:{reach_info.get('reachable_candidates', 0)}, bins:{reach_info.get('bins_nonempty', 0)})"
                        elif reach_info.get('max_reachable_dist', 0) > 0:
                            reach_str = f" reach=(max_dist:{reach_info['max_reachable_dist']:.2f}m, comp_size:{reach_info['reachable_component_size']})"
                        elif reach_info['checked'] > 0:
                            reach_str = f" reach=(checked:{reach_info['checked']}, rejected:{reach_info['rejected']}, nodes:{reach_info['nodes_explored']})"
                    
                    print(f"[RESET] start=({start_x:.2f},{start_y:.2f},{start_yaw:.1f}°) "
                          f"goal=({goal_x:.2f},{goal_y:.2f}) dist={dist:.2f}m "
                          f"clearance=(s:{clearance_start:.2f}m, g:{clearance_goal:.2f}m) tries={tries}{reach_str}", 
                          flush=True)
                
                # Fixed case capture: save first episode to JSON
                if self.episode_count == 1 and not self.fixed_case_saved and self.method_run_dir:
                    import json
                    import os
                    fixed_case_data = {
                        'start_x': float(start_x),
                        'start_y': float(start_y),
                        'start_yaw': float(start_yaw),
                        'goal_x': float(goal_x),
                        'goal_y': float(goal_y),
                        'map_name': 'small_house',
                        'sampler_mode': self.sampler_mode,
                        'distance_uniform': self.distance_uniform,
                        'timestamp': str(rospy.Time.now())
                    }
                    fixed_case_path = os.path.join(self.method_run_dir, 'fixed_case.json')
                    try:
                        with open(fixed_case_path, 'w') as f:
                            json.dump(fixed_case_data, f, indent=2)
                        self.fixed_case_saved = True
                        print(f"[RESET] Fixed case captured to: {fixed_case_path}", flush=True)
                        print(f"[RESET]   start=({start_x:.2f},{start_y:.2f},{start_yaw:.1f}°) goal=({goal_x:.2f},{goal_y:.2f})", flush=True)
                    except Exception as e:
                        print(f"[RESET] ERROR: Failed to save fixed case: {e}", flush=True)
                
                # Teleport robot to start position
                self.gazebo_helpers.set_robot_pose('turtlebot3_burger', start_x, start_y, start_yaw)
                
                # Longer settle time for physics stabilization + flush stale scans
                rospy.sleep(0.5)
                
                # Flush 2 stale scan messages (from before teleport)
                for _ in range(2):
                    try:
                        rospy.wait_for_message('scan', LaserScan, timeout=1.0)
                    except:
                        pass
                
                # Set goal position
                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y
                self.goal_position.position.z = 0.01
                
                # Build the targets (spawn goal marker)
                rospy.wait_for_service('/gazebo/spawn_sdf_model')
                try:
                    goal_urdf = open(goal_model_dir, "r").read()
                    target = SpawnModel
                    target.model_name = 'target'
                    target.model_xml = goal_urdf
                    
                    self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
                    
                except (rospy.ServiceException) as e:
                    pass
                
                # Get fresh scan after physics settled
                data = None
                while data is None:
                    try:
                        data = rospy.wait_for_message('scan', LaserScan, timeout=5)
                    except:
                        pass

                self.goal_distance = self.getGoalDistace()
                
                # Initialize reward function state
                self.reward_fn.reset(self.goal_distance)
                
                # Validate: check if first state triggers immediate collision
                state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
                
                # If no immediate collision, accept this reset
                if not done:
                    break
                
                # Collision detected in reset - retry with new sample
                if self.debug_sampler or attempt == max_reset_attempts - 1:
                    min_scan = min([r for r in data.ranges if not np.isnan(r) and r != float('Inf')])
                    print(f"[RESET WARNING] Attempt {attempt+1}/{max_reset_attempts}: Immediate collision detected (min_scan={min_scan:.3f}m < 0.2m). Resampling...", flush=True)
                
                # Delete the spawned target before retrying
                try:
                    self.del_model('target')
                except:
                    pass
            
            # Build final state
            state = [i / 3.5 for i in state]
        
        # Rectangle region sampler: sample from predefined regions with start != goal
        elif self.sampler_mode == 'rect_regions' and self.rect_sampler is not None:
            import math
            
            # Get current max distance for curriculum
            d_max_current = self.get_current_max_distance()
            d_min = self.curriculum_min_dist
            
            # Sample with distance filter (rejection sampling)
            max_attempts = 50
            result = None
            
            for attempt in range(max_attempts):
                candidate = self.rect_sampler.sample_start_goal()
                if candidate is None:
                    continue
                
                (sx, sy, syaw), (gx, gy), s_region_id, g_region_id = candidate
                dist = math.hypot(gx - sx, gy - sy)
                
                # Accept if within curriculum distance range
                if d_min <= dist <= d_max_current:
                    result = candidate
                    break
            
            # Fallback: if rejection sampling failed, place a fallback goal at
            # a guaranteed distance inside the curriculum range around the
            # sampled start. This avoids unexpectedly-large distances from
            # simply resampling a region-wide goal.
            if result is None:
                # Try to obtain a valid start to place the fallback goal around
                temp = self.rect_sampler.sample_start_goal()
                if temp is not None:
                    (sx, sy, syaw), (_, _), s_region_id, _ = temp
                else:
                    # Ultimate fallback start (map centre-ish)
                    sx, sy, syaw = 0.0, 0.0, 0.0
                    s_region_id = -1

                # Choose a fallback radius in the middle of [d_min, d_max_current]
                r = (d_min + d_max_current) / 2.0
                theta = random.random() * 2.0 * math.pi
                gx = sx + r * math.cos(theta)
                gy = sy + r * math.sin(theta)
                g_region_id = -1
            else:
                (sx, sy, syaw), (gx, gy), s_region_id, g_region_id = result
            
            # Log curriculum progress every 20 episodes
            if self.episode_count % 20 == 0:
                dist = math.hypot(gx - sx, gy - sy)
                print(f"[CURRICULUM ep{self.episode_count}] step={self.training_step} dist={dist:.2f}m range=[{d_min:.1f},{d_max_current:.1f}m]", flush=True)
            
            if self.debug_sampler:
                print(f"[RESET] start=R{s_region_id}({sx:.2f}, {sy:.2f}, {math.degrees(syaw):.1f}°) goal=R{g_region_id}({gx:.2f}, {gy:.2f})", flush=True)
            
            # Set robot pose using gazebo_reset_helpers
            success = self.gazebo_helpers.set_robot_pose('turtlebot3_burger', sx, sy, syaw, z=0.07)
            if not success:
                print("[RESET WARNING] set_robot_pose returned False", flush=True)
            
            # Set goal position
            self.goal_position.position.x = gx
            self.goal_position.position.y = gy
            self.goal_position.position.z = 0.01
            
            # Spawn goal marker
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'
                target.model_xml = goal_urdf
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            except (rospy.ServiceException) as e:
                print(f"[RESET WARNING] Failed to spawn goal marker: {e}", flush=True)
                pass
            
            # Wait for laser data after teleport
            rospy.sleep(0.1)
            data = None
            for _ in range(10):
                data = rospy.wait_for_message('scan', LaserScan, timeout=2)
                if data and len(data.ranges) > 0:
                    break
                rospy.sleep(0.05)
            
            if data is None:
                print("[RESET ERROR] Failed to get valid laser scan after rect-region reset", flush=True)
                state = [0.0] * 16
            else:
                # Calculate distances and angles
                self.goal_distance = self.getGoalDistace()
                
                # Initialize reward function
                self.reward_fn.reset(self.goal_distance)
                
                # Build state
                state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
                state = [i / 3.5 for i in state]
            
        else:
            # Default behavior: reset world (resets robot pose and physics)
            rospy.wait_for_service('/gazebo/reset_world')
            try:
                self.reset_world()
            except (rospy.ServiceException) as e:
                pass

            # Default goal sampling
            self.goal_position.position.x = random.uniform(-3.6, 3.6)
            self.goal_position.position.y = random.uniform(-3.6, 3.6)
            self.goal_position.position.z = 0.01
            while 1.7 <= self.goal_position.position.x <= 2.3 and -1.2 <= self.goal_position.position.y <= 1.2 \
                    or -2.3 <= self.goal_position.position.x <= -1.7 and -1.2 <= self.goal_position.position.y <= 1.2 \
                    or -1.2 <= self.goal_position.position.x <= 1.2 and 1.7 <= self.goal_position.position.y <= 2.3 \
                    or -1.2 <= self.goal_position.position.x <= 1.2 and -2.3 <= self.goal_position.position.y <= -1.7:
                self.goal_position.position.x = random.uniform(-3.6, 3.6)
                self.goal_position.position.y = random.uniform(-3.6, 3.6)
        
            # Build the targets (spawn goal marker)
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'
                target.model_xml = goal_urdf
                
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
                
            except (rospy.ServiceException) as e:
                pass
            
            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message('scan', LaserScan, timeout=5)
                except:
                    pass

            self.goal_distance = self.getGoalDistace()
            
            # Initialize reward function state
            self.reward_fn.reset(self.goal_distance)
            
            state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
            state = [i / 3.5 for i in state]
        
        # Uniform sampling: select 10 evenly-spaced samples from normalized scan
        L = len(state)
        indices = [int(i * L / 10) for i in range(10)]
        lidar_features = [state[idx] for idx in indices]
        
        # Build base state: 10 lidar + 2 past_action + 4 goalpose
        assert len(lidar_features) == 10, f"LiDAR features must be 10, got {len(lidar_features)}"
        base_state = lidar_features.copy()
        base_state.append(0)  # past_action[0]
        base_state.append(0)  # past_action[1]
        base_state = base_state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        # Assertion: Verify base state is exactly 16-d (10 lidar + 2 past_action + 4 goalpose)
        assert len(base_state) == 16, f"Base state must be 16-d, got {len(base_state)}"

        # Vision mode: return base state vector only (NO vision features appended here)
        # Image will be handled separately in PPO rollout

        return np.asarray(base_state)
