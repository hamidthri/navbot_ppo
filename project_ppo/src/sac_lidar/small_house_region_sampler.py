#!/usr/bin/env python3
"""
Region-based Position Sampler for Small House World

Samples robot spawn positions and goal positions from predefined rectangular regions.
Implements curriculum learning by gradually increasing the distance between robot and goal.
"""

import numpy as np
import random
from typing import Tuple, List, Dict


# Rectangular regions for open spaces in small house world
RECT_REGIONS = [
    {"id": 1, "name": "R1", "x": [2.78, 8.40], "y": [-4.40, -0.27]},
    {"id": 2, "name": "R2", "x": [-2.00, 2.78], "y": [-4.40, -3.46]},
    {"id": 3, "name": "R3", "x": [-2.02, 2.00], "y": [-0.27, 3.44]},
    {"id": 4, "name": "R4", "x": [-1.90, 4.29], "y": [4.71, 5.18]},  # Long tail of T
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


# Fixed robot spawn positions from 3 distant regions
# Selected from regions that are far apart: R1 (southeast), R5 (west), R9 (northeast)
ROBOT_SPAWN_POSITIONS = [
    {"name": "SpawnA_R1", "x": 5.5, "y": -2.5, "region_id": 1},   # Region R1
    {"name": "SpawnB_R5", "x": -5.5, "y": 0.0, "region_id": 5},   # Region R5
    {"name": "SpawnC_R9", "x": 7.0, "y": 2.5, "region_id": 9},    # Region R9
]


class SmallHouseRegionSampler:
    """
    Sampler for robot and goal positions in small house world.
    
    Features:
    - Samples from predefined rectangular regions
    - Fixed robot spawn positions (3 distant locations)
    - Curriculum learning: gradually increase distance between robot and goal
    - Ensures valid positions within region boundaries
    """
    
    def __init__(self, 
                 initial_distance=2.0, 
                 max_distance=18.0,
                 distance_increment=0.1,
                 increment_every_n_episodes=500):
        """
        Initialize sampler with curriculum learning parameters.
        
        Args:
            initial_distance (float): Starting distance between robot and goal (meters)
            max_distance (float): Maximum distance threshold (meters)
            distance_increment (float): Distance increase per curriculum step (meters)
            increment_every_n_episodes (int): Increase distance every N successful episodes
        """
        self.regions = RECT_REGIONS
        self.spawn_positions = ROBOT_SPAWN_POSITIONS
        
        # Curriculum learning parameters
        self.current_distance_threshold = initial_distance
        self.max_distance = max_distance
        self.distance_increment = distance_increment
        self.increment_every_n_episodes = increment_every_n_episodes
        
        # Episode tracking
        self.episode_count = 0
        self.success_count = 0
        
        print(f"[SmallHouseRegionSampler] Initialized with {len(self.regions)} regions")
        print(f"[SmallHouseRegionSampler] Robot spawn positions: {len(self.spawn_positions)} FIXED positions")
        print(f"[SmallHouseRegionSampler] The robot will ALWAYS start from one of these 3 positions randomly")
        print(f"[SmallHouseRegionSampler] Curriculum: start={initial_distance}m, max={max_distance}m, " +
              f"increment={distance_increment}m every {increment_every_n_episodes} successes")
        print(f"[SmallHouseRegionSampler] Will reach max distance after {int((max_distance - initial_distance) / distance_increment) * increment_every_n_episodes} successes")
    
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
        
        The robot will ALWAYS start from one of these 3 positions:
        - SpawnA: Region R1 (southeast area)
        - SpawnB: Region R5 (west area)  
        - SpawnC: Region R9 (northeast area)
        
        These positions are far apart to ensure diverse training scenarios.
        
        Returns:
            tuple: (x, y, name) coordinates and position name
        """
        spawn = random.choice(self.spawn_positions)
        return spawn["x"], spawn["y"], spawn["name"]
    
    def get_goal_position(self, robot_x: float, robot_y: float, 
                         min_distance: float = None) -> Tuple[float, float, str]:
        """
        Sample a goal position from regions based on current curriculum distance.
        
        Args:
            robot_x (float): Robot's current x position
            robot_y (float): Robot's current y position
            min_distance (float): Minimum distance from robot (default: current curriculum threshold)
            
        Returns:
            tuple: (x, y, region_name) coordinates and region name
        """
        if min_distance is None:
            min_distance = self.current_distance_threshold
        
        # Use a tighter range for more uniform sampling
        # Instead of [min_distance, max_distance], use [min_distance, min_distance + 6.0]
        # This ensures goals are within a reasonable range from the robot
        max_distance_for_sample = min(min_distance + 6.0, self.max_distance)
        
        max_attempts = 100
        for attempt in range(max_attempts):
            # Sample random region
            region = random.choice(self.regions)
            
            # Sample position within region
            goal_x, goal_y = self.sample_position_from_region(region)
            
            # Check distance
            distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
            
            # Accept if within tighter curriculum range
            if min_distance <= distance <= max_distance_for_sample:
                return goal_x, goal_y, region["name"]
        
        # Fallback: try with wider range
        for attempt in range(max_attempts):
            region = random.choice(self.regions)
            goal_x, goal_y = self.sample_position_from_region(region)
            distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
            
            if min_distance <= distance <= self.max_distance:
                return goal_x, goal_y, region["name"]
        
        # Last fallback: return any valid position
        print(f"[Warning] Could not find goal within distance range after {max_attempts*2} attempts. " +
              f"Using fallback position. Min: {min_distance:.1f}m, Max: {max_distance_for_sample:.1f}m")
        region = random.choice(self.regions)
        goal_x, goal_y = self.sample_position_from_region(region)
        return goal_x, goal_y, region["name"]
    
    def update_curriculum(self, success: bool):
        """
        Update curriculum learning progress.
        
        Args:
            success (bool): Whether the episode was successful
        """
        self.episode_count += 1
        
        if success:
            self.success_count += 1
            
            # Check if we should increase difficulty
            if (self.success_count % self.increment_every_n_episodes == 0 and 
                self.current_distance_threshold < self.max_distance):
                
                old_threshold = self.current_distance_threshold
                self.current_distance_threshold = min(
                    self.current_distance_threshold + self.distance_increment,
                    self.max_distance
                )
                
                print(f"\n[Curriculum] Distance threshold increased: " +
                      f"{old_threshold:.1f}m → {self.current_distance_threshold:.1f}m " +
                      f"(after {self.success_count} successes)\n")
    
    def get_current_distance_threshold(self) -> float:
        """Get current curriculum distance threshold."""
        return self.current_distance_threshold
    
    def get_stats(self) -> Dict:
        """Get sampler statistics."""
        return {
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "current_distance_threshold": self.current_distance_threshold,
            "max_distance": self.max_distance,
        }
    
    def reset_curriculum(self, initial_distance: float = 2.0):
        """Reset curriculum to initial state."""
        self.current_distance_threshold = initial_distance
        self.episode_count = 0
        self.success_count = 0
        print(f"[Curriculum] Reset to initial distance: {initial_distance}m")


# Convenience functions
def sample_robot_position() -> Tuple[float, float]:
    """Sample a robot spawn position (for backward compatibility)."""
    sampler = SmallHouseRegionSampler()
    x, y, _ = sampler.get_robot_spawn_position()
    return x, y


def sample_goal_position(robot_x: float, robot_y: float, 
                        distance_threshold: float = 2.0) -> Tuple[float, float]:
    """Sample a goal position (for backward compatibility)."""
    sampler = SmallHouseRegionSampler(initial_distance=distance_threshold)
    x, y, _ = sampler.get_goal_position(robot_x, robot_y)
    return x, y


if __name__ == "__main__":
    # Test the sampler
    print("Testing SmallHouseRegionSampler...\n")
    
    sampler = SmallHouseRegionSampler(initial_distance=2.0, max_distance=15.0)
    
    # Test robot spawn positions
    print("=== Robot Spawn Positions ===")
    for i in range(5):
        x, y, name = sampler.get_robot_spawn_position()
        print(f"  Spawn {i+1}: ({x:.2f}, {y:.2f}) - {name}")
    
    # Test goal sampling
    print("\n=== Goal Sampling (from each spawn) ===")
    for spawn in ROBOT_SPAWN_POSITIONS:
        robot_x, robot_y = spawn["x"], spawn["y"]
        print(f"\nRobot at {spawn['name']}: ({robot_x:.2f}, {robot_y:.2f})")
        
        for i in range(3):
            goal_x, goal_y, region = sampler.get_goal_position(robot_x, robot_y)
            distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
            print(f"  Goal {i+1}: ({goal_x:.2f}, {goal_y:.2f}) in {region}, distance={distance:.2f}m")
    
    # Test curriculum learning
    print("\n=== Curriculum Learning ===")
    sampler_curriculum = SmallHouseRegionSampler(
        initial_distance=2.0, 
        distance_increment=1.0, 
        increment_every_n_episodes=5
    )
    
    for i in range(20):
        success = (i % 2 == 0)  # Simulate 50% success rate
        sampler_curriculum.update_curriculum(success)
        
        if i % 5 == 0:
            stats = sampler_curriculum.get_stats()
            print(f"Episode {stats['episode_count']}: " +
                  f"Successes={stats['success_count']}, " +
                  f"Distance={stats['current_distance_threshold']:.1f}m")
    
    print("\n✓ SmallHouseRegionSampler tests completed!")
