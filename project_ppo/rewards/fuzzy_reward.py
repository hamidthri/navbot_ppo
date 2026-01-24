#!/usr/bin/env python3
"""
Fuzzy reward implementation based on the paper specification.

Implements binary fuzzy logic with:
- 3 inputs: de, dy, dtheta (improvement checks via switch function)
- 8 rules (Good/Bad -> Perfect/Good/Bad/Terrible)
- Gamma obstacle factor (1 - min_lidar_clipped)
- Mamdani inference + centroid defuzzification
"""

import numpy as np
import yaml
import os


def switch(x):
    """Binary switch function: +99 if improving (x>0), -99 otherwise."""
    return 99.0 if x > 0 else -99.0


class FuzzyReward:
    """
    Paper-spec fuzzy reward with sequential-step improvement checks.
    
    Inputs:
        - de: distance improvement (switch applied)
        - dy: yaw improvement (switch applied) 
        - dtheta: heading improvement (switch applied)
        - min_lidar: minimum laser range for gamma
    
    Rules (8 binary rules):
        1. G G G -> Perfect
        2. G G B -> Good
        3. G B G -> Good
        4. G B B -> Bad
        5. B G G -> Good
        6. B G B -> Bad
        7. B B G -> Bad
        8. B B B -> Terrible
    
    Gamma: 1 - clip(min_lidar, 0, lidar_clip_m) / lidar_clip_m
    Output: gamma * fuzzy_output
    """
    
    def __init__(self, config_path=None):
        """Initialize fuzzy system from config."""
        # Default config
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                '../config/fuzzy_reward.yaml'
            )
        
        # Load config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Store params
        self.lidar_clip_m = cfg['gamma']['lidar_clip_m']
        
        # Define membership functions (binary: Good if x > 0, Bad if x <= 0)
        self.mf = cfg['membership_functions']
        
        # Define rule base (8 rules)
        self.rules = cfg['rules']
        
        # Output MF centers
        self.output_mf_centers = cfg['output_centers']
        
        # Previous state for sequential improvement
        self.prev_distance = None
        self.prev_yaw = None
        self.prev_rel_theta = None
    
    def reset(self, initial_distance, initial_yaw, initial_rel_theta):
        """Reset for new episode - store initial state."""
        self.prev_distance = initial_distance
        self.prev_yaw = initial_yaw
        self.prev_rel_theta = initial_rel_theta
    
    def _membership_good(self, x):
        """Good membership: 1 if x > 0, else 0."""
        return 1.0 if x > 0 else 0.0
    
    def _membership_bad(self, x):
        """Bad membership: 1 if x <= 0, else 0."""
        return 1.0 if x <= 0 else 0.0
    
    def _compute_gamma(self, min_lidar):
        """Gamma obstacle factor: 1 - clip(min_lidar) / lidar_clip."""
        clipped = np.clip(min_lidar, 0.0, self.lidar_clip_m)
        gamma = 1.0 - clipped / self.lidar_clip_m
        return gamma
    
    def _evaluate_rules(self, de, dy, dtheta):
        """
        Evaluate all 8 rules using Mamdani inference.
        
        Returns:
            aggregated output membership (dict: center -> activation)
        """
        # Compute input memberships
        de_good = self._membership_good(de)
        de_bad = self._membership_bad(de)
        dy_good = self._membership_good(dy)
        dy_bad = self._membership_bad(dy)
        dtheta_good = self._membership_good(dtheta)
        dtheta_bad = self._membership_bad(dtheta)
        
        # Rule activations (AND = min)
        activations = {}
        
        # Rule 1: G G G -> Perfect
        activations['Perfect'] = min(de_good, dy_good, dtheta_good)
        
        # Rule 2: G G B -> Good
        act2 = min(de_good, dy_good, dtheta_bad)
        
        # Rule 3: G B G -> Good
        act3 = min(de_good, dy_bad, dtheta_good)
        
        # Rule 4: G B B -> Bad
        act4 = min(de_good, dy_bad, dtheta_bad)
        
        # Rule 5: B G G -> Good
        act5 = min(de_bad, dy_good, dtheta_good)
        
        # Rule 6: B G B -> Bad
        act6 = min(de_bad, dy_good, dtheta_bad)
        
        # Rule 7: B B G -> Bad
        act7 = min(de_bad, dy_bad, dtheta_good)
        
        # Rule 8: B B B -> Terrible
        activations['Terrible'] = min(de_bad, dy_bad, dtheta_bad)
        
        # Aggregate Good rules (max)
        activations['Good'] = max(act2, act3, act5)
        
        # Aggregate Bad rules (max)
        activations['Bad'] = max(act4, act6, act7)
        
        return activations
    
    def _defuzzify(self, activations):
        """Centroid defuzzification."""
        numerator = 0.0
        denominator = 0.0
        
        for label, activation in activations.items():
            if activation > 0:
                center = self.output_mf_centers[label]
                numerator += center * activation
                denominator += activation
        
        if denominator == 0:
            return 0.0  # Fallback if no rules fire
        
        return numerator / denominator
    
    def compute(self, current_distance, current_yaw, current_rel_theta, 
                min_lidar, done, arrive):
        """
        Compute fuzzy reward.
        
        Args:
            current_distance: distance to goal
            current_yaw: robot yaw
            current_rel_theta: relative angle to goal
            min_lidar: minimum laser range
            done: collision flag
            arrive: goal reached flag
        
        Returns:
            reward: fuzzy reward value
        """
        # Terminal states use fixed rewards (same as legacy)
        if done:
            return -100.0
        
        if arrive:
            return 120.0
        
        # Compute improvement deltas
        de = self.prev_distance - current_distance  # positive = getting closer
        
        # Yaw improvement: reduction in angular error
        # (simplified: check if yaw changed in direction of target)
        dy = abs(self.prev_rel_theta - self.prev_yaw) - abs(current_rel_theta - current_yaw)
        
        # Heading improvement: check if rel_theta decreased
        dtheta = abs(self.prev_rel_theta) - abs(current_rel_theta)
        
        # Apply switch function
        de_sw = switch(de)
        dy_sw = switch(dy)
        dtheta_sw = switch(dtheta)
        
        # Evaluate fuzzy rules
        activations = self._evaluate_rules(de_sw, dy_sw, dtheta_sw)
        
        # Defuzzify
        fuzzy_output = self._defuzzify(activations)
        
        # Compute gamma
        gamma = self._compute_gamma(min_lidar)
        
        # Final reward
        reward = gamma * fuzzy_output
        
        # Update previous state
        self.prev_distance = current_distance
        self.prev_yaw = current_yaw
        self.prev_rel_theta = current_rel_theta
        
        return reward
