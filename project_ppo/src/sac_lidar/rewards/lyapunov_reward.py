#!/usr/bin/env python3
"""
Lyapunov-based Reward Function for SAC Lidar Navigation

This reward function uses Control Lyapunov Function (CLF) and 
Control Barrier Function (CBF) principles for navigation with safety.

Environment dimensions: 17m x 10m

Reward components:
1. CLF (Control Lyapunov Function): Encourages convergence to goal
   r_clf = 30 * (d_prev² - d_curr²) / √d_curr
   
2. CBF (Control Barrier Function): Ensures safety (obstacle avoidance)
   h = min_laser_distance - safety_margin
   if h < 0:         r_cbf = -150  (collision zone)
   elif h < 0.3:     r_cbf = -50 * (1 - h/0.3)²  (danger zone, smooth penalty)
   else:             r_cbf = 0  (safe zone)

r_total = r_clf + r_cbf
"""
import math
import numpy as np


class LyapunovReward:
    """
    Lyapunov CLF+CBF based reward function.
    
    Uses Control Lyapunov Function for goal convergence and
    Control Barrier Function for safety enforcement.
    """
    
    def __init__(self,
                 # CLF parameters
                 clf_scale: float = 30.0,
                 
                 # CBF parameters  
                 # Environment collision threshold: min_range = 0.2m
                 # When min_laser < 0.2, done=True (episode ends)
                 # So safety_margin should match this threshold
                 safety_margin: float = 0.2,   # same as environment's min_range
                 danger_zone: float = 0.3,     # smooth penalty zone: 0.2 to 0.5m
                 collision_penalty: float = -100.0,  # matched to legacy collision penalty
                 danger_scale: float = -30.0,  # reduced sensitivity in danger zone
                 
                 # Terminal rewards (matched to legacy for fair comparison)
                 arrival_bonus: float = 120.0,
                 
                 # Environment dimensions (17m x 10m)
                 env_width: float = 17.0,
                 env_height: float = 10.0,
                 
                 # Numerical stability
                 min_distance: float = 0.1,
                 
                 **kwargs):
        """
        Initialize Lyapunov reward function.
        
        Args:
            clf_scale: Scale factor for CLF reward (default 30)
            safety_margin: Minimum safe distance from obstacles (default 0.2m)
            danger_zone: Distance where smooth penalty starts (default 0.3m)
            collision_penalty: Penalty for being in collision zone (default -150)
            danger_scale: Scale for danger zone penalty (default -50)
            arrival_bonus: Bonus for reaching goal
            env_width: Environment width in meters (17m)
            env_height: Environment height in meters (10m)
            min_distance: Minimum distance for numerical stability
        """
        self.clf_scale = clf_scale
        self.safety_margin = safety_margin
        self.danger_zone = danger_zone
        self.collision_penalty = collision_penalty
        self.danger_scale = danger_scale
        self.arrival_bonus = arrival_bonus
        self.env_width = env_width
        self.env_height = env_height
        self.min_distance = min_distance
        
        # Diagonal distance for normalization
        self.diagonal_dis = math.sqrt(env_width**2 + env_height**2)
        
        print(f"[LyapunovReward] Initialized with:")
        print(f"  CLF scale: {clf_scale}")
        print(f"  Safety margin: {safety_margin}m")
        print(f"  Danger zone: {danger_zone}m")
        print(f"  Collision penalty: {collision_penalty}")
        print(f"  Danger scale: {danger_scale}")
        print(f"  Arrival bonus: {arrival_bonus}")
        print(f"  Environment: {env_width}m x {env_height}m")
    
    def _compute_clf_reward(self, current_distance: float, past_distance: float) -> float:
        """
        Compute Control Lyapunov Function reward.
        
        CLF encourages Lyapunov function decrease (convergence to goal).
        V(x) = d² (squared distance to goal)
        r_clf = scale * (V_prev - V_curr) / √d_curr
             = scale * (d_prev² - d_curr²) / √d_curr
        
        Args:
            current_distance: Current distance to goal
            past_distance: Previous distance to goal
            
        Returns:
            CLF reward value
        """
        # Ensure numerical stability
        d_curr = max(current_distance, self.min_distance)
        d_prev = max(past_distance, self.min_distance)
        
        # Lyapunov function values (squared distance)
        V_curr = d_curr ** 2
        V_prev = d_prev ** 2
        
        # CLF reward: encourage Lyapunov decrease, normalized by sqrt(d_curr)
        # This gives higher reward when close to goal (more sensitive)
        r_clf = self.clf_scale * (V_prev - V_curr) / math.sqrt(d_curr)
        
        return r_clf
    
    def _compute_cbf_reward(self, min_laser_distance: float) -> float:
        """
        Compute Control Barrier Function reward.
        
        CBF enforces safety through a smooth barrier function.
        h = min_laser - safety_margin
        
        if h < 0:         r_cbf = collision_penalty  (in collision zone)
        elif h < danger:  r_cbf = danger_scale * (1 - h/danger)²  (smooth penalty)
        else:             r_cbf = 0  (safe)
        
        Args:
            min_laser_distance: Minimum distance reading from lidar
            
        Returns:
            CBF reward value
        """
        # Barrier function value
        h = min_laser_distance - self.safety_margin
        
        if h < 0:
            # In collision zone - maximum penalty
            r_cbf = self.collision_penalty
        elif h < self.danger_zone:
            # In danger zone - smooth quadratic penalty
            # Penalty decreases from danger_scale to 0 as h goes from 0 to danger_zone
            normalized_h = h / self.danger_zone
            r_cbf = self.danger_scale * (1 - normalized_h) ** 2
        else:
            # Safe zone - no penalty
            r_cbf = 0.0
        
        return r_cbf
    
    def compute_reward(self,
                       current_distance: float,
                       past_distance: float,
                       min_laser_distance: float,
                       heading_error: float = None,
                       done: bool = False,
                       arrive: bool = False) -> tuple:
        """
        Compute the Lyapunov CLF+CBF reward.
        
        Args:
            current_distance: Current distance to goal
            past_distance: Previous distance to goal
            min_laser_distance: Minimum laser scan reading
            heading_error: Angle difference to goal in degrees (optional, for future use)
            done: Whether episode ended due to collision
            arrive: Whether robot reached goal
            
        Returns:
            tuple: (reward, info_dict)
        """
        # Compute CLF reward (goal convergence)
        r_clf = self._compute_clf_reward(current_distance, past_distance)
        
        # Compute CBF reward (safety)
        r_cbf = self._compute_cbf_reward(min_laser_distance)
        
        # Total reward
        reward = r_clf + r_cbf
        
        info = {
            'clf_reward': r_clf,
            'cbf_reward': r_cbf,
            'collision_penalty': 0.0,
            'arrival_bonus': 0.0,
            'min_laser': min_laser_distance,
            'barrier_h': min_laser_distance - self.safety_margin,
            'reward_type': 'lyapunov'
        }
        
        # Override for terminal states
        if done:
            # Collision already penalized by CBF, but add explicit marker
            info['collision_penalty'] = r_cbf
            # Don't override reward - CBF already handles collision penalty
        
        if arrive:
            # Add arrival bonus on top of current reward
            reward = self.arrival_bonus
            info['arrival_bonus'] = self.arrival_bonus
            info['clf_reward'] = 0.0
            info['cbf_reward'] = 0.0
        
        return reward, info
    
    def reset(self):
        """Reset any internal state (not needed for this reward)."""
        pass
