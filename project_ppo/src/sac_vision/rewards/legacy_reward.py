#!/usr/bin/env python3
"""
Legacy Reward Function for SAC Lidar Navigation

This is the original reward function used in the environment.
It uses distance-based reward with fixed collision and arrival bonuses.

Reward components:
- Distance reward: 500 * (d_prev - d_curr)  [progress toward goal]
- Collision penalty: -100
- Arrival bonus: +120
"""
import math


class LegacyReward:
    """
    Legacy distance-based reward function.
    
    This reward encourages the robot to move toward the goal
    with simple linear distance improvement reward.
    """
    
    def __init__(self, 
                 distance_scale: float = 500.0,
                 collision_penalty: float = -100.0,
                 arrival_bonus: float = 120.0,
                 **kwargs):
        """
        Initialize legacy reward function.
        
        Args:
            distance_scale: Multiplier for distance improvement
            collision_penalty: Reward when robot collides
            arrival_bonus: Reward when robot reaches goal
        """
        self.distance_scale = distance_scale
        self.collision_penalty = collision_penalty
        self.arrival_bonus = arrival_bonus
        
        print(f"[LegacyReward] Initialized with:")
        print(f"  Distance scale: {distance_scale}")
        print(f"  Collision penalty: {collision_penalty}")
        print(f"  Arrival bonus: {arrival_bonus}")
    
    def compute_reward(self, 
                       current_distance: float,
                       past_distance: float,
                       min_laser_distance: float,
                       heading_error: float = None,
                       done: bool = False,
                       arrive: bool = False) -> tuple:
        """
        Compute the legacy reward.
        
        Args:
            current_distance: Current distance to goal
            past_distance: Previous distance to goal
            min_laser_distance: Minimum laser scan reading (for collision detection)
            heading_error: Angle difference to goal (not used in legacy)
            done: Whether episode ended (collision)
            arrive: Whether robot reached goal
            
        Returns:
            tuple: (reward, info_dict)
        """
        # Distance improvement reward
        distance_rate = past_distance - current_distance
        reward = self.distance_scale * distance_rate
        
        info = {
            'distance_reward': reward,
            'collision_penalty': 0.0,
            'arrival_bonus': 0.0,
            'reward_type': 'legacy'
        }
        
        # Collision penalty
        if done:
            reward = self.collision_penalty
            info['collision_penalty'] = self.collision_penalty
            info['distance_reward'] = 0.0
        
        # Arrival bonus
        if arrive:
            reward = self.arrival_bonus
            info['arrival_bonus'] = self.arrival_bonus
            info['distance_reward'] = 0.0
        
        return reward, info
    
    def reset(self):
        """Reset any internal state (not needed for legacy reward)."""
        pass
