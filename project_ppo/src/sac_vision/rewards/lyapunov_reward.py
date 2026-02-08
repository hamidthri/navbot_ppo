#!/usr/bin/env python3
"""
Optimized Directional CBF Reward for 10-Ray Forward-Facing Lidar

Laser configuration: 10 beams from -90° to 90°
Angles: [-90, -70, -50, -30, -10, 10, 30, 50, 70, 90] degrees

This reward function uses:
1. Control Lyapunov Function (CLF) for goal convergence
2. Directional Control Barrier Function (CBF) for intelligent obstacle avoidance
   - Weights obstacles by alignment with goal direction
   - Solves "stuck in corridor" problem
   - Maintains safety guarantees

Key innovation: Since lidar is already forward-facing, we weight rays
by how aligned they are with the goal direction using cos²(angle_diff).
"""

import math
import numpy as np


class DirectionalCBFReward:
    """
    Directional CLF+CBF reward optimized for 10-ray forward lidar.
    
    Solves corridor navigation by distinguishing forward obstacles from side constraints.
    """
    
    def __init__(self,
                 # CLF parameters
                 clf_scale: float = 30.0,
                 
                 # Directional CBF parameters  
                 safety_margin: float = 0.2,      # Min safe distance (matches env collision threshold)
                 danger_zone: float = 0.3,        # Smooth penalty zone
                 collision_penalty: float = -100.0,
                 danger_scale: float = -30.0,
                 
                 # Directional weighting
                 min_forward_weight: float = 0.05,  # Lower threshold since we have fewer rays
                 directional_sharpness: float = 2.0,  # Exponent for cos weighting (1=linear, 2=quadratic)
                 
                 # Terminal rewards
                 arrival_bonus: float = 120.0,
                 
                 # Environment
                 env_width: float = 17.0,
                 env_height: float = 10.0,
                 
                 # Numerical stability
                 min_distance: float = 0.1,
                 
                 **kwargs):
        """
        Initialize directional CBF reward.
        
        Args:
            clf_scale: Scale for CLF goal attraction (default 30)
            safety_margin: Collision threshold (default 0.2m, matches env)
            danger_zone: Distance where smooth penalty starts (default 0.3m)
            collision_penalty: Hard collision penalty (default -100)
            danger_scale: Scale for proximity penalty (default -30)
            min_forward_weight: Minimum weight to consider obstacle (default 0.05)
            directional_sharpness: Exponent for directional weighting (default 2.0)
            arrival_bonus: Reward for reaching goal (default 120)
        """
        self.clf_scale = clf_scale
        self.safety_margin = safety_margin
        self.danger_zone = danger_zone
        self.collision_penalty = collision_penalty
        self.danger_scale = danger_scale
        self.min_forward_weight = min_forward_weight
        self.directional_sharpness = directional_sharpness
        self.arrival_bonus = arrival_bonus
        self.env_width = env_width
        self.env_height = env_height
        self.min_distance = min_distance
        
        # Lidar configuration: 10 rays from -90° to 90°
        # Angles: -90, -70, -50, -30, -10, 10, 30, 50, 70, 90
        self.n_rays = 10
        self.laser_angles = np.radians(np.linspace(-90, 90, self.n_rays))
        
        self.diagonal_dis = math.sqrt(env_width**2 + env_height**2)
        
        print(f"[DirectionalCBF] Initialized with:")
        print(f"  CLF scale: {clf_scale}")
        print(f"  Safety margin: {safety_margin}m")
        print(f"  Danger zone: {danger_zone}m")
        print(f"  Directional sharpness: {directional_sharpness}")
        print(f"  Min forward weight: {min_forward_weight}")
        print(f"  Lidar: {self.n_rays} rays from -90° to 90°")
        print(f"  Environment: {env_width}m x {env_height}m")
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _compute_clf_reward(self, current_distance: float, past_distance: float) -> float:
        """
        Compute Control Lyapunov Function reward.
        
        CLF encourages goal convergence through Lyapunov function decrease.
        V(x) = d² (squared distance to goal)
        r_clf = scale * (V_prev - V_curr) / √d_curr
        
        This creates:
        - Stronger pull when far from goal
        - Gentler approach when near goal (natural deceleration)
        - Guaranteed convergence if V always decreases
        """
        d_curr = max(current_distance, self.min_distance)
        d_prev = max(past_distance, self.min_distance)
        
        V_curr = d_curr ** 2
        V_prev = d_prev ** 2
        
        # Normalized CLF reward
        r_clf = self.clf_scale * (V_prev - V_curr) / math.sqrt(d_curr)
        
        return r_clf
    
    def _compute_goal_direction(self, robot_x, robot_y, goal_x, goal_y):
        """Compute angle from robot to goal in robot frame"""
        return math.atan2(goal_y - robot_y, goal_x - robot_x)
    
    def _compute_directional_cbf_reward(self, 
                                        laser_scan: list,
                                        robot_x: float, 
                                        robot_y: float,
                                        robot_yaw_deg: float,
                                        goal_x: float, 
                                        goal_y: float) -> tuple:
        """
        Compute directional CBF reward with cosine weighting.
        
        Key idea: Weight each laser ray by how aligned it is with goal direction.
        - Front obstacles (aligned with goal): high weight → strong penalty if close
        - Side obstacles (perpendicular to goal): low weight → weak or no penalty
        
        Formula:
        1. Compute goal direction in robot frame
        2. For each ray: weight = max(0, cos(ray_angle - goal_angle))^sharpness
        3. Effective distance = weighted minimum
        4. Standard CBF penalty on effective distance
        
        Args:
            laser_scan: List of 10 laser distances
            robot_x, robot_y: Robot position
            robot_yaw_deg: Robot heading in degrees
            goal_x, goal_y: Goal position
            
        Returns:
            tuple: (cbf_reward, effective_distance, debug_info)
        """
        # Convert inputs to numpy arrays
        laser_scan = np.array(laser_scan)
        
        # Compute goal direction in world frame
        goal_angle_world = self._compute_goal_direction(robot_x, robot_y, goal_x, goal_y)
        
        # Convert robot yaw to radians
        robot_yaw_rad = math.radians(robot_yaw_deg)
        
        # Goal direction in robot frame (relative to robot's forward direction)
        goal_angle_robot = self._normalize_angle(goal_angle_world - robot_yaw_rad)
        
        # Compute angle difference between each laser ray and goal direction
        # laser_angles are already in robot frame: 0° = forward, +90° = left, -90° = right
        angle_diffs = np.array([
            self._normalize_angle(laser_angle - goal_angle_robot)
            for laser_angle in self.laser_angles
        ])
        
        # Cosine weighting: cos^sharpness(angle_diff)
        # sharpness=2 (default): quadratic falloff
        # sharpness=1: linear falloff
        # Higher sharpness = narrower focus on forward direction
        weights = np.maximum(0, np.cos(angle_diffs)) ** self.directional_sharpness
        
        # Filter out rays with very low weights
        significant_mask = weights >= self.min_forward_weight
        
        if not np.any(significant_mask):
            # All obstacles are to the side (shouldn't happen with forward lidar, but safety check)
            # Fallback to omnidirectional minimum
            effective_distance = np.min(laser_scan)
            debug_info = {
                'used_rays': 0,
                'goal_angle_robot_deg': math.degrees(goal_angle_robot),
                'weights': weights.tolist(),
                'fallback': True
            }
        else:
            # Compute weighted effective distance
            # Use "weighted minimum" approach: prioritize close obstacles in forward direction
            filtered_distances = laser_scan[significant_mask]
            filtered_weights = weights[significant_mask]
            
            # Threat score: weight / distance
            # Higher score = more threatening (close AND in forward direction)
            threat_scores = filtered_weights / (filtered_distances + 0.01)
            
            # Most threatening obstacle determines effective distance
            max_threat_idx = np.argmax(threat_scores)
            effective_distance = filtered_distances[max_threat_idx]
            
            debug_info = {
                'used_rays': int(np.sum(significant_mask)),
                'goal_angle_robot_deg': math.degrees(goal_angle_robot),
                'weights': weights.tolist(),
                'effective_ray_idx': int(np.where(significant_mask)[0][max_threat_idx]),
                'fallback': False
            }
        
        # Compute CBF penalty using effective distance
        h = effective_distance - self.safety_margin
        
        if h < 0:
            # Collision zone
            r_cbf = self.collision_penalty
        elif h < self.danger_zone:
            # Danger zone: smooth quadratic penalty
            normalized_h = h / self.danger_zone
            r_cbf = self.danger_scale * (1 - normalized_h) ** 2
        else:
            # Safe zone
            r_cbf = 0.0
        
        debug_info['barrier_h'] = h
        debug_info['effective_distance'] = effective_distance
        debug_info['min_laser'] = float(np.min(laser_scan))
        
        return r_cbf, effective_distance, debug_info
    
    def compute_reward(self,
                       current_distance: float,
                       past_distance: float,
                       laser_scan: list,
                       robot_x: float,
                       robot_y: float,
                       robot_yaw_deg: float,
                       goal_x: float,
                       goal_y: float,
                       heading_error: float = None,
                       done: bool = False,
                       arrive: bool = False) -> tuple:
        """
        Compute total directional CLF+CBF reward.
        
        Args:
            current_distance: Current distance to goal
            past_distance: Previous distance to goal
            laser_scan: List of 10 laser readings
            robot_x, robot_y: Robot position
            robot_yaw_deg: Robot heading in degrees
            goal_x, goal_y: Goal position
            heading_error: Heading error in degrees (optional, for logging)
            done: Episode ended (collision)
            arrive: Goal reached
            
        Returns:
            tuple: (reward, info_dict)
        """
        # CLF reward (goal attraction)
        r_clf = self._compute_clf_reward(current_distance, past_distance)
        
        # Directional CBF reward (intelligent obstacle avoidance)
        r_cbf, effective_dist, debug_info = self._compute_directional_cbf_reward(
            laser_scan, robot_x, robot_y, robot_yaw_deg, goal_x, goal_y
        )
        
        # Total reward
        reward = r_clf + r_cbf
        
        # Build info dict
        info = {
            'clf_reward': r_clf,
            'cbf_reward': r_cbf,
            'effective_distance': effective_dist,
            'min_laser': debug_info['min_laser'],
            'barrier_h': debug_info['barrier_h'],
            'used_rays': debug_info['used_rays'],
            'goal_angle_robot_deg': debug_info.get('goal_angle_robot_deg', 0),
            'collision_penalty': 0.0,
            'arrival_bonus': 0.0,
            'reward_type': 'directional_cbf'
        }
        
        # Terminal states
        if done:
            # Collision already penalized by CBF
            info['collision_penalty'] = r_cbf
        
        if arrive:
            # Override with arrival bonus
            reward = self.arrival_bonus
            info['arrival_bonus'] = self.arrival_bonus
            info['clf_reward'] = 0.0
            info['cbf_reward'] = 0.0
        
        return reward, info
    
    def reset(self):
        """Reset any internal state (none needed for this reward)"""
        pass


# ============================================================================
# Legacy Reward Function (for comparison)
# ============================================================================

class LegacyReward:
    """Original distance-based reward function for comparison."""
    
    def __init__(self, **kwargs):
        self.past_distance = 0.
        print("[LegacyReward] Initialized")
    
    def compute_reward(self, current_distance, past_distance, laser_scan=None,
                       robot_x=None, robot_y=None, robot_yaw_deg=None,
                       goal_x=None, goal_y=None, heading_error=None,
                       done=False, arrive=False):
        """Compute legacy distance-based reward"""
        distance_rate = (past_distance - current_distance)
        reward = 500. * distance_rate
        
        info = {
            'clf_reward': 0.0,
            'cbf_reward': 0.0,
            'min_laser': min(laser_scan) if laser_scan else 3.5,
            'reward_type': 'legacy'
        }
        
        if done:
            reward = -100.
        
        if arrive:
            reward = 120.
        
        return reward, info
    
    def reset(self):
        pass