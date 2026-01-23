"""
Map-based start/goal sampler using OccupancyGrid + DistanceTransform.
Ensures collision-free sampling with configurable clearance thresholds.
Includes reachability checking via BFS to ensure goal is accessible from start.
"""

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt
from PIL import Image
import math
import os
from collections import deque


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
                 max_tries=100,
                 check_reachability=True,
                 max_bfs_nodes=10000,
                 distance_uniform=False,
                 distance_bins=10,
                 min_bin_candidates=10):
        """
        Args:
            map_yaml_path: Path to ROS map YAML file
            clearance_start: Minimum clearance for start positions (meters)
            clearance_goal: Minimum clearance for goal positions (meters)
            min_distance: Minimum Euclidean distance between start and goal (meters)
            max_distance: Maximum Euclidean distance between start and goal (meters)
            max_tries: Maximum sampling attempts before giving up
            check_reachability: Whether to verify goal is reachable from start via BFS
            max_bfs_nodes: Maximum BFS nodes to explore (safety limit)
            distance_uniform: If True, sample uniformly across distance bins (not spatial)
            distance_bins: Number of equal-width distance bins for distance_uniform mode
            min_bin_candidates: Minimum candidates per bin to be eligible for sampling
        """
        self.map_yaml_path = map_yaml_path
        self.clearance_start = clearance_start
        self.clearance_goal = clearance_goal
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_tries = max_tries
        self.check_reachability = check_reachability
        self.max_bfs_nodes = max_bfs_nodes
        self.distance_uniform = distance_uniform
        self.distance_bins = distance_bins
        self.min_bin_candidates = min_bin_candidates
        
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
    
    def _is_reachable(self, start_row, start_col, goal_row, goal_col):
        """
        Check if goal cell is reachable from start cell using BFS on free cells.
        
        Args:
            start_row, start_col: Start cell coordinates
            goal_row, goal_col: Goal cell coordinates
            
        Returns:
            (reachable: bool, nodes_explored: int)
        """
        if not self.check_reachability:
            return True, 0
        
        # Check bounds
        if not (0 <= start_row < self.height and 0 <= start_col < self.width):
            return False, 0
        if not (0 <= goal_row < self.height and 0 <= goal_col < self.width):
            return False, 0
        
        # Check if cells are free
        if self.free_map[start_row, start_col] == 0:
            return False, 0
        if self.free_map[goal_row, goal_col] == 0:
            return False, 0
        
        # BFS
        queue = deque([(start_row, start_col)])
        visited = set()
        visited.add((start_row, start_col))
        nodes_explored = 0
        
        # 8-connected neighbors for better connectivity
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 4-connected
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonals
        ]
        
        while queue and nodes_explored < self.max_bfs_nodes:
            row, col = queue.popleft()
            nodes_explored += 1
            
            # Check if we reached the goal
            if row == goal_row and col == goal_col:
                return True, nodes_explored
            
            # Explore neighbors
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if not (0 <= new_row < self.height and 0 <= new_col < self.width):
                    continue
                
                # Check if already visited
                if (new_row, new_col) in visited:
                    continue
                
                # Check if free cell
                if self.free_map[new_row, new_col] == 0:
                    continue
                
                visited.add((new_row, new_col))
                queue.append((new_row, new_col))
        
        return False, nodes_explored
    
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
    
    def sample_goal(self, start_xy, debug=False):
        """
        Sample a valid goal position given start position.
        Pure uniform sampling over valid cells (no bias).
        
        Args:
            start_xy: (x, y) tuple of start position
            debug: Whether to return reachability debug info
            
        Returns:
            goal_xy: (x, y) tuple
            tries: number of sampling attempts
            accepted: always True if returns successfully
            reachability_info: dict with BFS stats (only if debug=True)
        """
        start_x, start_y = start_xy
        start_row, start_col = self._world_to_pixel(start_x, start_y)
        
        reachability_info = {'checked': 0, 'rejected': 0, 'nodes_explored': 0}
        
        for tries in range(1, self.max_tries + 1):
            # Uniform sampling from valid goal cells
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
                # Check reachability
                if self.check_reachability:
                    reachable, nodes = self._is_reachable(start_row, start_col, row, col)
                    reachability_info['checked'] += 1
                    reachability_info['nodes_explored'] += nodes
                    
                    if not reachable:
                        reachability_info['rejected'] += 1
                        continue  # Reject unreachable goal
                
                if debug:
                    return (goal_x, goal_y), tries, True, reachability_info
                return (goal_x, goal_y), tries, True
        
        raise RuntimeError(
            f"Failed to sample valid goal after {self.max_tries} tries "
            f"(start=({start_x:.2f}, {start_y:.2f}), "
            f"distance range={self.min_distance}-{self.max_distance}m)"
        )
    
    def sample_goal_distance_uniform(self, start_xy, debug=False):
        """
        Sample goal with uniform distribution across DISTANCE bins (not spatial).
        
        Algorithm:
        1. Compute all reachable goal candidates from start (BFS once)
        2. Bin candidates by distance
        3. Sample bin uniformly from non-empty bins with >= min_bin_candidates
        4. Sample goal uniformly from chosen bin
        
        Returns same format as sample_goal().
        """
        start_x, start_y = start_xy
        start_row, start_col = self._world_to_pixel(start_x, start_y)
        
        reachability_info = {
            'checked': 0,
            'rejected': 0, 
            'nodes_explored': 0,
            'reachable_candidates': 0,
            'bins_nonempty': 0,
            'chosen_bin': -1
        }
        
        # Step 1: BFS to find all reachable cells from start
        if not self.check_reachability:
            # If reachability disabled, use all valid goal cells
            reachable_cells = set(map(tuple, self.valid_goal_cells))
        else:
            queue = deque([(start_row, start_col)])
            visited = set()
            visited.add((start_row, start_col))
            
            directions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
            
            nodes_explored = 0
            while queue and nodes_explored < self.max_bfs_nodes:
                row, col = queue.popleft()
                nodes_explored += 1
                
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    
                    if not (0 <= new_row < self.height and 0 <= new_col < self.width):
                        continue
                    if (new_row, new_col) in visited:
                        continue
                    if self.free_map[new_row, new_col] == 0:
                        continue
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))
            
            reachable_cells = visited
            reachability_info['nodes_explored'] = nodes_explored
        
        # Step 2: Filter reachable cells by goal clearance and distance
        # Convert valid_goal_cells to set for O(1) lookup
        valid_goal_set = set(map(tuple, self.valid_goal_cells))
        
        candidates = []
        for row, col in reachable_cells:
            # Check if it's a valid goal cell (clearance check)
            if (row, col) not in valid_goal_set:
                continue
            
            goal_x, goal_y = self._pixel_to_world(row, col)
            dist = math.hypot(goal_x - start_x, goal_y - start_y)
            
            if self.min_distance <= dist <= self.max_distance:
                candidates.append((row, col, dist))
        
        reachability_info['reachable_candidates'] = len(candidates)
        
        if len(candidates) == 0:
            raise RuntimeError(
                f"No reachable goal candidates found "
                f"(start=({start_x:.2f}, {start_y:.2f}), reachable_cells={len(reachable_cells)})"
            )
        
        # Step 3: Bin candidates by distance (vectorized)
        candidates_array = np.array(candidates)
        rows = candidates_array[:, 0].astype(int)
        cols = candidates_array[:, 1].astype(int)
        dists = candidates_array[:, 2]
        
        # Create bins
        bin_width = (self.max_distance - self.min_distance) / self.distance_bins
        bin_edges = np.linspace(self.min_distance, self.max_distance, self.distance_bins + 1)
        bin_indices = np.digitize(dists, bin_edges) - 1  # 0-indexed bins
        bin_indices = np.clip(bin_indices, 0, self.distance_bins - 1)  # Handle edge cases
        
        # Group candidates by bin
        bins = [[] for _ in range(self.distance_bins)]
        for i, bin_idx in enumerate(bin_indices):
            bins[bin_idx].append(i)  # Store index into candidates array
        
        # Count non-empty bins with sufficient candidates
        eligible_bins = [b for b in range(self.distance_bins) if len(bins[b]) >= self.min_bin_candidates]
        reachability_info['bins_nonempty'] = len([b for b in bins if len(b) > 0])
        
        if len(eligible_bins) == 0:
            # Fallback: use non-empty bins even if below min_bin_candidates
            eligible_bins = [b for b in range(self.distance_bins) if len(bins[b]) > 0]
            if len(eligible_bins) == 0:
                raise RuntimeError(f"No non-empty bins (candidates={len(candidates)})")
        
        # Step 4: Sample bin uniformly from eligible bins
        chosen_bin = np.random.choice(eligible_bins)
        reachability_info['chosen_bin'] = chosen_bin
        
        # Step 5: Sample goal uniformly from chosen bin
        candidate_idx = np.random.choice(bins[chosen_bin])
        goal_row, goal_col, goal_dist = candidates[candidate_idx]
        goal_x, goal_y = self._pixel_to_world(int(goal_row), int(goal_col))
        
        if debug:
            return (goal_x, goal_y), 1, True, reachability_info
        return (goal_x, goal_y), 1, True
    
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
