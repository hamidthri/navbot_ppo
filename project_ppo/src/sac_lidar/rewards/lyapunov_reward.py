#!/usr/bin/env python3
"""
Directional CBF Reward for 10-Ray Lidar Navigation

Drop-in replacement for lyapunov_reward.py with directional obstacle awareness.
Optimized for 10 laser rays from -90° to 90°.
"""
import math
import numpy as np


class DirectionalCBFReward:
    """
    Directional CLF+CBF reward function.
    Solves corridor navigation by weighting obstacles based on goal direction.
    """
    
    def __init__(self,
                 clf_scale: float = 30.0,
                 safety_margin: float = 0.2,
                 danger_zone: float = 0.15,
                 collision_penalty: float = -100.0,
                 danger_scale: float = -10.0,
                 arrival_bonus: float = 120.0,
                 min_forward_weight: float = 0.05,
                 directional_sharpness: float = 2.0,
                 env_width: float = 17.0,
                 env_height: float = 10.0,
                 min_distance: float = 0.1,
                 **kwargs):
        
        self.clf_scale = clf_scale
        self.safety_margin = safety_margin
        self.danger_zone = danger_zone
        self.collision_penalty = collision_penalty
        self.danger_scale = danger_scale
        self.arrival_bonus = arrival_bonus
        self.min_forward_weight = min_forward_weight
        self.directional_sharpness = directional_sharpness
        self.env_width = env_width
        self.env_height = env_height
        self.min_distance = min_distance
        
        # 10 rays from -90° to 90°
        self.n_rays = 10
        self.laser_angles = np.radians(np.linspace(-90, 90, self.n_rays))
        
        self.diagonal_dis = math.sqrt(env_width**2 + env_height**2)
        
        print(f"[DirectionalCBF] Initialized:")
        print(f"  CLF: {clf_scale}, Safety: {safety_margin}m, Danger: {danger_zone}m")
        print(f"  Directional: sharpness={directional_sharpness}, min_weight={min_forward_weight}")
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _compute_clf_reward(self, current_distance: float, past_distance: float) -> float:
        """CLF: V(x) = d², reward = scale * (V_prev - V_curr) / √d_curr"""
        d_curr = max(current_distance, self.min_distance)
        d_prev = max(past_distance, self.min_distance)
        
        V_curr = d_curr ** 2
        V_prev = d_prev ** 2
        
        r_clf = self.clf_scale * (V_prev - V_curr) / math.sqrt(d_curr)
        return r_clf
    
    def _compute_directional_cbf(self, laser_scan, robot_x, robot_y, robot_yaw_deg, goal_x, goal_y):
        """Directional CBF: weight obstacles by alignment with goal direction"""
        laser_scan = np.array(laser_scan)
        
        # Goal direction in world frame
        goal_angle_world = math.atan2(goal_y - robot_y, goal_x - robot_x)
        robot_yaw_rad = math.radians(robot_yaw_deg)
        goal_angle_robot = self._normalize_angle(goal_angle_world - robot_yaw_rad)
        
        # Compute directional weights: cos^sharpness(angle_diff)
        angle_diffs = np.array([
            self._normalize_angle(laser_angle - goal_angle_robot)
            for laser_angle in self.laser_angles
        ])
        weights = np.maximum(0, np.cos(angle_diffs)) ** self.directional_sharpness
        
        # Filter significant rays
        significant_mask = weights >= self.min_forward_weight
        
        if not np.any(significant_mask):
            # Fallback to omnidirectional
            effective_distance = np.min(laser_scan)
        else:
            # Weighted minimum: prioritize close obstacles in forward direction
            filtered_distances = laser_scan[significant_mask]
            filtered_weights = weights[significant_mask]
            threat_scores = filtered_weights / (filtered_distances + 0.01)
            max_threat_idx = np.argmax(threat_scores)
            effective_distance = filtered_distances[max_threat_idx]
        
        # CBF penalty
        h = effective_distance - self.safety_margin
        
        if h < 0:
            r_cbf = self.collision_penalty
        elif h < self.danger_zone:
            normalized_h = h / self.danger_zone
            r_cbf = self.danger_scale * (1 - normalized_h) ** 2
        else:
            r_cbf = 0.0
        
        return r_cbf, effective_distance
    
    def compute_reward(self,
                       current_distance: float,
                       past_distance: float,
                       min_laser_distance: float = None,
                       laser_scan: list = None,
                       robot_x: float = None,
                       robot_y: float = None,
                       robot_yaw_deg: float = None,
                       goal_x: float = None,
                       goal_y: float = None,
                       heading_error: float = None,
                       done: bool = False,
                       arrive: bool = False) -> tuple:
        """
        Compute directional CLF+CBF reward.
        
        Backward compatible: works with both full info (directional) and min_laser only.
        """
        # CLF reward
        r_clf = self._compute_clf_reward(current_distance, past_distance)
        
        # CBF reward
        if laser_scan is not None and robot_x is not None and goal_x is not None:
            # Directional mode
            r_cbf, effective_dist = self._compute_directional_cbf(
                laser_scan, robot_x, robot_y, robot_yaw_deg, goal_x, goal_y
            )
        else:
            # Fallback to simple omnidirectional mode
            h = min_laser_distance - self.safety_margin
            if h < 0:
                r_cbf = self.collision_penalty
            elif h < self.danger_zone:
                normalized_h = h / self.danger_zone
                r_cbf = self.danger_scale * (1 - normalized_h) ** 2
            else:
                r_cbf = 0.0
            effective_dist = min_laser_distance
        
        # Total reward
        reward = r_clf + r_cbf
        
        info = {
            'clf_reward': r_clf,
            'cbf_reward': r_cbf,
            'collision_penalty': 0.0,
            'arrival_bonus': 0.0,
            'min_laser': min_laser_distance if min_laser_distance else effective_dist,
            'barrier_h': effective_dist - self.safety_margin,
            'reward_type': 'directional_cbf'
        }
        
        if done:
            info['collision_penalty'] = r_cbf
        
        if arrive:
            reward = self.arrival_bonus
            info['arrival_bonus'] = self.arrival_bonus
            info['clf_reward'] = 0.0
            info['cbf_reward'] = 0.0
        
        return reward, info
    
    def reset(self):
        pass