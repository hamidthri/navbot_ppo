"""
Map-based start/goal sampler using OccupancyGrid + DistanceTransform.
Ensures collision-free sampling with configurable clearance thresholds.
"""

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt
from PIL import Image
import math
import os


class MapGoalSampler:
    """
    Sample start poses and goal positions from an occupancy grid map
    using distance transform to ensure minimum clearance from obstacles.
    """
    
    def __init__(self, map_yaml_path, 
                 clearance_start=0.70, 
                 clearance_goal=0.45,
                 min_distance=2.5,
                 max_distance=8.0,
                 max_tries=100):
        """
        Args:
            map_yaml_path: Path to ROS map YAML file
            clearance_start: Minimum clearance for start positions (meters)
            clearance_goal: Minimum clearance for goal positions (meters)
            min_distance: Minimum Euclidean distance between start and goal (meters)
            max_distance: Maximum Euclidean distance between start and goal (meters)
            max_tries: Maximum sampling attempts before giving up
        """
        self.map_yaml_path = map_yaml_path
        self.clearance_start = clearance_start
        self.clearance_goal = clearance_goal
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_tries = max_tries
        
        # Load map and compute distance transforms
        self._load_map(map_yaml_path)
        self._compute_distance_transforms()
        self._precompute_valid_cells()
        
    def _load_map(self, yaml_path):
        """Load occupancy grid from ROS YAML+PGM/PNG format."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Map YAML not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            map_info = yaml.safe_load(f)
        
        # Load image
        map_dir = os.path.dirname(yaml_path)
        image_path = os.path.join(map_dir, map_info['image'])
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Map image not found: {image_path}")
        
        # Load as grayscale
        img = Image.open(image_path).convert('L')
        self.map_image = np.array(img)
        
        # Store map metadata
        self.resolution = map_info['resolution']  # meters/pixel
        self.origin = map_info['origin']  # [x, y, theta]
        self.height, self.width = self.map_image.shape
        
        # Convert to binary: free (255) vs occupied (0)
        # ROS convention: 0=free, 100=occupied, -1=unknown
        # PGM: 254=free, 0=occupied, 205=unknown
        self.free_map = (self.map_image > 250).astype(np.uint8) * 255
        
    def _compute_distance_transforms(self):
        """Compute distance to nearest obstacle for all free cells."""
        # Distance transform: for each free pixel, compute distance to nearest occupied pixel
        # scipy's distance_transform_edt works on binary images (True = background)
        binary_free = (self.free_map > 0).astype(bool)
        self.distance_transform = distance_transform_edt(binary_free)
        
        # Convert from pixels to meters
        self.distance_transform_m = self.distance_transform * self.resolution
        
    def _precompute_valid_cells(self):
        """Precompute valid cells for start and goal based on clearance."""
        # Start positions: clearance >= clearance_start
        self.valid_start_cells = np.argwhere(
            self.distance_transform_m >= self.clearance_start
        )
        
        # Goal positions: clearance >= clearance_goal
        self.valid_goal_cells = np.argwhere(
            self.distance_transform_m >= self.clearance_goal
        )
        
        if len(self.valid_start_cells) == 0:
            raise ValueError(f"No valid start cells with clearance >= {self.clearance_start}m")
        if len(self.valid_goal_cells) == 0:
            raise ValueError(f"No valid goal cells with clearance >= {self.clearance_goal}m")
        
    def _pixel_to_world(self, row, col):
        """Convert pixel coordinates to world coordinates (x, y in meters)."""
        # Note: row corresponds to y-axis, col to x-axis
        x = col * self.resolution + self.origin[0]
        y = row * self.resolution + self.origin[1]
        return x, y
    
    def _world_to_pixel(self, x, y):
        """Convert world coordinates to pixel coordinates."""
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return row, col
    
    def get_clearance(self, x, y):
        """Get clearance (distance to nearest obstacle) at world position (x, y)."""
        row, col = self._world_to_pixel(x, y)
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.distance_transform_m[row, col]
        return 0.0
    
    def sample_start(self):
        """
        Sample a valid start position with required clearance.
        
        Returns:
            start_pose: (x, y, yaw) tuple
            tries: number of sampling attempts
        """
        for tries in range(1, self.max_tries + 1):
            # Sample from valid start cells
            idx = np.random.randint(len(self.valid_start_cells))
            row, col = self.valid_start_cells[idx]
            x, y = self._pixel_to_world(row, col)
            
            # Hard check: verify clearance (safety verification)
            actual_clearance = self.distance_transform_m[row, col]
            if actual_clearance < self.clearance_start:
                continue  # Reject and retry
            
            # Random yaw orientation
            yaw = np.random.uniform(0, 360)
            
            return (x, y, yaw), tries
        
        raise RuntimeError(f"Failed to sample valid start after {self.max_tries} tries")
    
    def sample_goal(self, start_xy):
        """
        Sample a valid goal position given start position.
        
        Args:
            start_xy: (x, y) tuple of start position
            
        Returns:
            goal_xy: (x, y) tuple
            tries: number of sampling attempts
            accepted: always True if returns successfully
        """
        start_x, start_y = start_xy
        
        for tries in range(1, self.max_tries + 1):
            # Sample from valid goal cells
            idx = np.random.randint(len(self.valid_goal_cells))
            row, col = self.valid_goal_cells[idx]
            goal_x, goal_y = self._pixel_to_world(row, col)
            
            # Hard check: verify clearance (safety verification)
            actual_clearance = self.distance_transform_m[row, col]
            if actual_clearance < self.clearance_goal:
                continue  # Reject and retry
            
            # Check distance constraint
            dist = math.hypot(goal_x - start_x, goal_y - start_y)
            
            if self.min_distance <= dist <= self.max_distance:
                return (goal_x, goal_y), tries, True
        
        raise RuntimeError(
            f"Failed to sample valid goal after {self.max_tries} tries "
            f"(start=({start_x:.2f}, {start_y:.2f}), "
            f"distance range={self.min_distance}-{self.max_distance}m)"
        )
    
    def sample_start_and_goal(self):
        """
        Sample both start pose and goal position.
        
        Returns:
            start_pose: (x, y, yaw) tuple
            goal_xy: (x, y) tuple
            total_tries: total number of sampling attempts
        """
        start_pose, start_tries = self.sample_start()
        goal_xy, goal_tries, _ = self.sample_goal(start_pose[:2])
        
        return start_pose, goal_xy, start_tries + goal_tries
