#!/usr/bin/env python3
"""
Region-based Position Sampler for Small House World

Samples robot spawn positions (3 fixed points) and goal positions (random in regions).
Uses probabilistic curriculum learning for goal distance distribution.
"""

import numpy as np
import random
from typing import Tuple, Dict


# Rectangular regions for open spaces in small house world
RECT_REGIONS = [
    {"id": 1, "name": "R1", "x": [2.78, 8.40], "y": [-4.40, -0.27]},
    {"id": 2, "name": "R2", "x": [-2.00, 2.78], "y": [-4.40, -3.46]},
    {"id": 3, "name": "R3", "x": [-2.02, 2.00], "y": [-0.27, 3.44]},
    {"id": 4, "name": "R4", "x": [-1.90, 4.29], "y": [4.71, 5.18]},
    {"id": 5, "name": "R5", "x": [-8.21, -3.03], "y": [-1.05, 1.00]},
    {"id": 6, "name": "R6", "x": [-8.85, -3.18], "y": [-3.60, -1.84]},
    {"id": 7, "name": "R7", "x": [-4.96, -3.90], "y": [-0.78, 2.40]},
    {"id": 8, "name": "R8", "x": [-8.13, -7.43], "y": [1.16, 2.34]},
    {"id": 9, "name": "R9", "x": [5.22, 8.79], "y": [2.25, 2.85]},
    {"id": 10, "name": "R10", "x": [7.65, 8.79], "y": [0.00, 2.25]},
    {"id": 11, "name": "R11", "x": [0.28, 0.57], "y": [-1.79, -0.74]},
    {"id": 12, "name": "R12", "x": [-6.42, -3.53], "y": [-5.02, -3.88]},
    {"id": 13, "name": "R13", "x": [2.81, 3.49], "y": [1.78, 2.88]},
]


# Fixed robot spawn positions (3 distant locations)
ROBOT_SPAWN_POSITIONS = [
    {"name": "SpawnA_R1", "x": 5.5, "y": -2.5, "region_id": 1},   # Region R1 (southeast)
    {"name": "SpawnB_R5", "x": -5.5, "y": 0.0, "region_id": 5},   # Region R5 (west)
    {"name": "SpawnC_R9", "x": 7.0, "y": 2.5, "region_id": 9},    # Region R9 (northeast)
]


class SmallHouseRegionSampler:
    """
    Sampler for robot and goal positions in small house world.
    
    Features:
    - Robot: 3 FIXED spawn positions (always same locations)
    - Goals: Random positions within predefined regions
    - Probabilistic curriculum: 50% close (1-3m), 30% medium (3-6m), 20% far (6+m)
    """
    
    def __init__(self, initial_distance=2.0, max_distance=18.0, distance_increment=0.1, increment_every_n_episodes=500):
        """
        Initialize sampler with probabilistic curriculum.
        
        Args:
            initial_distance: Starting distance for curriculum (default: 2.0m)  
            max_distance: Maximum distance for curriculum (default: 18.0m)
            distance_increment: Distance increase per curriculum step (default: 0.1m)
            increment_every_n_episodes: Episodes between curriculum updates (default: 500)
        """
        self.regions = RECT_REGIONS
        self.spawn_positions = ROBOT_SPAWN_POSITIONS
        
        # Curriculum parameters (for compatibility, but not used in probabilistic mode)
        self.initial_distance = initial_distance
        self.max_distance = max_distance
        self.distance_increment = distance_increment
        self.increment_every_n_episodes = increment_every_n_episodes
        
        # Probabilistic curriculum parameters
        # 50% chance: 1-3m, 30% chance: 3-6m, 20% chance: 6+m
        self.distance_ranges = [
            {"min": 1.0, "max": 3.0, "prob": 0.5, "name": "close"},
            {"min": 3.0, "max": 6.0, "prob": 0.3, "name": "medium"},
            {"min": 6.0, "max": 18.0, "prob": 0.2, "name": "far"}
        ]
        
        # Statistics tracking
        self.episode_count = 0
        self.goal_distance_stats = {"close": 0, "medium": 0, "far": 0}
        
        print(f"[SmallHouseRegionSampler] Initialized with {len(self.regions)} regions")
        print(f"[SmallHouseRegionSampler] Robot spawn: {len(self.spawn_positions)} FIXED positions")
        print(f"[SmallHouseRegionSampler] Goal sampling: RANDOM within regions")
        print(f"[SmallHouseRegionSampler] Probabilistic curriculum:")
        print(f"  - 50% probability: 1-3m (close)")
        print(f"  - 30% probability: 3-6m (medium)")
        print(f"  - 20% probability: 6+m (far)")
    
    def sample_position_from_region(self, region: Dict) -> Tuple[float, float]:
        """
        Sample a random position within a rectangular region.
        
        Args:
            region (dict): Region definition with 'x' and 'y' bounds
            
        Returns:
            tuple: (x, y) coordinates
        """
        x = np.random.uniform(region["x"][0], region["x"][1])
        y = np.random.uniform(region["y"][0], region["y"][1])
        return x, y
    
    def get_robot_spawn_position(self) -> Tuple[float, float, str]:
        """
        Get a random robot spawn position from the 3 FIXED locations.
        
        Returns:
            tuple: (x, y, name) coordinates and position name
        """
        spawn = random.choice(self.spawn_positions)
        return spawn["x"], spawn["y"], spawn["name"]
    
    def select_distance_range(self) -> Dict:
        """
        Select distance range based on probabilities.
        50% close (1-3m), 30% medium (3-6m), 20% far (6+m)
        
        Returns:
            dict: Selected distance range
        """
        # Generate random number [0, 1)
        rand = random.random()
        
        # Cumulative probability selection
        cumulative = 0.0
        for range_config in self.distance_ranges:
            cumulative += range_config["prob"]
            if rand < cumulative:
                return range_config
        
        # Fallback (should never happen)
        return self.distance_ranges[-1]
    
    def get_goal_position(self, robot_x: float, robot_y: float) -> Tuple[float, float, str]:
        """
        Sample a goal position randomly within regions.
        Distance from robot follows probabilistic curriculum.
        
        Args:
            robot_x (float): Robot's current x position
            robot_y (float): Robot's current y position
            
        Returns:
            tuple: (x, y, region_name) coordinates and region name
        """
        # Select distance range based on probabilities
        distance_range = self.select_distance_range()
        min_distance = distance_range["min"]
        max_distance = distance_range["max"]
        range_name = distance_range["name"]
        
        # Track statistics
        self.goal_distance_stats[range_name] += 1
        
        # Try to find a goal within the selected distance range
        max_attempts = 100
        for attempt in range(max_attempts):
            # Sample random region
            region = random.choice(self.regions)
            
            # Sample position within region
            goal_x, goal_y = self.sample_position_from_region(region)
            
            # Check distance
            distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
            
            # Accept if within selected range
            if min_distance <= distance <= max_distance:
                return goal_x, goal_y, region["name"]
        
        # Fallback: relax constraints and try again
        for attempt in range(max_attempts):
            region = random.choice(self.regions)
            goal_x, goal_y = self.sample_position_from_region(region)
            distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
            
            # Accept if at least minimum distance (relax max constraint)
            if distance >= min_distance:
                return goal_x, goal_y, region["name"]
        
        # Last fallback: return any valid position
        region = random.choice(self.regions)
        goal_x, goal_y = self.sample_position_from_region(region)
        return goal_x, goal_y, region["name"]
    
    def update_curriculum(self, success: bool):
        """
        Update episode counter (curriculum is probabilistic, no changes needed).
        
        Args:
            success (bool): Whether the episode was successful
        """
        self.episode_count += 1
        
        # Print statistics every 100 episodes
        if self.episode_count % 100 == 0:
            total = sum(self.goal_distance_stats.values())
            if total > 0:
                close_pct = 100 * self.goal_distance_stats["close"] / total
                medium_pct = 100 * self.goal_distance_stats["medium"] / total
                far_pct = 100 * self.goal_distance_stats["far"] / total
                
                print(f"\n[Curriculum Stats @ {self.episode_count} episodes]")
                print(f"  Goal distance distribution:")
                print(f"    Close (1-3m):  {close_pct:.1f}% ({self.goal_distance_stats['close']}/{total})")
                print(f"    Medium (3-6m): {medium_pct:.1f}% ({self.goal_distance_stats['medium']}/{total})")
                print(f"    Far (6+m):     {far_pct:.1f}% ({self.goal_distance_stats['far']}/{total})")
                print()
    
    def get_stats(self) -> Dict:
        """Get sampler statistics."""
        return {
            "episode_count": self.episode_count,
            "goal_distance_stats": self.goal_distance_stats.copy(),
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.episode_count = 0
        self.goal_distance_stats = {"close": 0, "medium": 0, "far": 0}
        print(f"[Curriculum] Statistics reset")


if __name__ == "__main__":
    # Test the sampler
    print("Testing SmallHouseRegionSampler with Probabilistic Curriculum...\n")
    
    sampler = SmallHouseRegionSampler()
    
    # Test robot spawn positions
    print("=== Robot Spawn Positions (FIXED) ===")
    for i in range(5):
        x, y, name = sampler.get_robot_spawn_position()
        print(f"  Spawn {i+1}: ({x:.2f}, {y:.2f}) - {name}")
    
    # Test goal sampling from each spawn
    print("\n=== Goal Sampling (Random in Regions) ===")
    for spawn in ROBOT_SPAWN_POSITIONS:
        robot_x, robot_y = spawn["x"], spawn["y"]
        print(f"\nRobot at {spawn['name']}: ({robot_x:.2f}, {robot_y:.2f})")
        
        for i in range(5):
            goal_x, goal_y, region = sampler.get_goal_position(robot_x, robot_y)
            distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
            
            # Determine range
            if distance < 3.0:
                range_name = "close"
            elif distance < 6.0:
                range_name = "medium"
            else:
                range_name = "far"
            
            # print(f"  Goal {i+1}: ({goal_x:.2f}, {goal_y:.2f}) in {region}, "
            #       f"distance={distance:.2f}m ({range_name})")
    
    # Test probabilistic distribution
    print("\n=== Testing Probabilistic Distribution (1000 samples) ===")
    sampler_test = SmallHouseRegionSampler()
    
    robot_x, robot_y = 5.5, -2.5  # Fixed robot position
    distance_samples = []
    
    for i in range(1000):
        goal_x, goal_y, region = sampler_test.get_goal_position(robot_x, robot_y)
        distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
        distance_samples.append(distance)
        sampler_test.update_curriculum(success=True)
    
    # Analyze distribution
    close_count = sum(1 for d in distance_samples if 1.0 <= d < 3.0)
    medium_count = sum(1 for d in distance_samples if 3.0 <= d < 6.0)
    far_count = sum(1 for d in distance_samples if d >= 6.0)
    
    print(f"\nActual distribution (out of 1000 samples):")
    print(f"  Close (1-3m):  {close_count/10:.1f}% (expected ~50%)")
    print(f"  Medium (3-6m): {medium_count/10:.1f}% (expected ~30%)")
    print(f"  Far (6+m):     {far_count/10:.1f}% (expected ~20%)")
    
    print("\n✓ SmallHouseRegionSampler tests completed!")