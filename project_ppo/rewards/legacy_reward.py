#!/usr/bin/env python3
"""
Legacy reward function extracted from environment_new.py.
Preserved verbatim with no logic changes.
"""

class LegacyReward:
    """
    Original reward computation from the baseline environment.
    
    Reward components:
    - Distance progress: 500 * (past_distance - current_distance)
    - Collision: -100
    - Goal arrival: +120
    """
    
    def __init__(self):
        self.past_distance = 0.0
    
    def reset(self, initial_distance):
        """Reset state for new episode."""
        self.past_distance = initial_distance
    
    def compute(self, current_distance, done, arrive):
        """
        Compute reward based on current state.
        
        Args:
            current_distance: Euclidean distance to goal
            done: True if collision occurred
            arrive: True if goal reached
        
        Returns:
            reward: scalar reward value
        """
        distance_rate = (self.past_distance - current_distance)
        reward = 500.0 * distance_rate
        self.past_distance = current_distance
        
        if done:
            reward = -100.0
        
        if arrive:
            reward = 120.0
        
        return reward
