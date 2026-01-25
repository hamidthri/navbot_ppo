"""
Rectangle-region based start/goal sampler.
Samples from 13 predefined rectangular regions with constraint:
start and goal must be from different regions.
"""

import numpy as np
import random


# 13 Rectangle Regions (axis-aligned)
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


class RectRegionSampler:
    """
    Sample start/goal poses from predefined rectangular regions.
    Enforces: start_region != goal_region
    
    SHAPED DISTANCE SAMPLING:
    - Samples goal distance r from shaped distribution (bins + probabilities)
    - Then samples goal at that distance from start
    - Ensures goals fall within valid regions and constraints
    """
    
    def __init__(self, robot_radius=0.105, clearance_margin=0.30, max_tries=100,
                 use_shaped_distance=True, distance_bins=None, bin_probs=None,
                 min_goal_dist=2.0, max_goal_dist=12.0):
        """
        Args:
            robot_radius: Robot radius in meters
            clearance_margin: Safety margin from rectangle boundaries
            max_tries: Maximum attempts before giving up
            use_shaped_distance: Enable shaped distance sampling
            distance_bins: List of (min, max) tuples for distance bins
                          Default: [(2,6), (6,10), (10,12)]
            bin_probs: List of probabilities for each bin (must sum to 1.0)
                      Default: [0.70, 0.25, 0.05]
            min_goal_dist: Hard minimum goal distance (m)
            max_goal_dist: Hard maximum goal distance (m)
        """
        self.robot_radius = robot_radius
        self.clearance_margin = clearance_margin
        self.total_clearance = robot_radius + clearance_margin
        self.max_tries = max_tries
        self.regions = RECT_REGIONS
        
        # Shaped distance sampling configuration
        self.use_shaped_distance = use_shaped_distance
        self.min_goal_dist = min_goal_dist
        self.max_goal_dist = max_goal_dist
        
        if distance_bins is None:
            self.distance_bins = [(2.0, 6.0), (6.0, 10.0), (10.0, 12.0)]
        else:
            self.distance_bins = distance_bins
            
        if bin_probs is None:
            self.bin_probs = [0.70, 0.25, 0.05]
        else:
            self.bin_probs = bin_probs
            
        # Validate configuration
        assert len(self.distance_bins) == len(self.bin_probs), "Bins and probs must match"
        assert abs(sum(self.bin_probs) - 1.0) < 1e-6, "Probabilities must sum to 1.0"
        
        # Statistics
        self.stats = {
            "start_region_counts": {i: 0 for i in range(1, 14)},
            "goal_region_counts": {i: 0 for i in range(1, 14)},
            "total_samples": 0,
            "rejections": {"boundary": 0, "same_region": 0, "max_tries": 0, "shaped_distance": 0},
            "distances": [],  # Track actual start-goal distances
        }
    
    def _sample_point_in_region(self, region):
        """
        Sample a random point within a region, respecting clearance.
        
        Args:
            region: Dict with 'x' and 'y' bounds
            
        Returns:
            (x, y) tuple or None if region too small for clearance
        """
        x_min, x_max = region["x"]
        y_min, y_max = region["y"]
        
        # Apply clearance margin
        x_min_safe = x_min + self.total_clearance
        x_max_safe = x_max - self.total_clearance
        y_min_safe = y_min + self.total_clearance
        y_max_safe = y_max - self.total_clearance
        
        # Check if region is large enough
        if x_max_safe <= x_min_safe or y_max_safe <= y_min_safe:
            return None
        
        x = np.random.uniform(x_min_safe, x_max_safe)
        y = np.random.uniform(y_min_safe, y_max_safe)
        return (x, y)
    
    def _point_in_region(self, x, y, region):
        """
        Check if a point is within a region (respecting clearance).
        
        Args:
            x, y: Point coordinates
            region: Dict with 'x' and 'y' bounds
            
        Returns:
            bool: True if point is safely within region
        """
        x_min, x_max = region["x"]
        y_min, y_max = region["y"]
        
        x_min_safe = x_min + self.total_clearance
        x_max_safe = x_max - self.total_clearance
        y_min_safe = y_min + self.total_clearance
        y_max_safe = y_max - self.total_clearance
        
        return (x_min_safe <= x <= x_max_safe) and (y_min_safe <= y <= y_max_safe)
    
    def _sample_distance_from_bins(self):
        """
        Sample a target distance from shaped distribution.
        
        Returns:
            float: Target distance in meters
        """
        # Sample a bin
        bin_idx = np.random.choice(len(self.distance_bins), p=self.bin_probs)
        bin_min, bin_max = self.distance_bins[bin_idx]
        
        # Sample uniformly within bin
        r = np.random.uniform(bin_min, bin_max)
        
        # Clamp to hard limits
        r = np.clip(r, self.min_goal_dist, self.max_goal_dist)
        
        return r
    
    def _sample_goal_at_distance(self, start_x, start_y, start_region_id, target_distance):
        """
        Sample a goal at approximately target_distance from start.
        
        Args:
            start_x, start_y: Start position
            start_region_id: Start region ID
            target_distance: Target distance in meters
            
        Returns:
            tuple: ((goal_x, goal_y), goal_region_id) or None if failed
        """
        # Try to find a valid goal at the target distance
        for attempt in range(self.max_tries):
            # Sample random angle
            theta = np.random.uniform(0, 2 * np.pi)
            
            # Propose goal at target distance
            goal_x = start_x + target_distance * np.cos(theta)
            goal_y = start_y + target_distance * np.sin(theta)
            
            # Check if goal is in a valid region (different from start)
            for region in self.regions:
                if region["id"] == start_region_id:
                    continue  # Skip start region
                    
                if self._point_in_region(goal_x, goal_y, region):
                    return (goal_x, goal_y), region["id"]
        
        return None
    
    def sample_start_goal(self):
        """
        Sample start pose and goal position from different regions.
        Uses shaped distance sampling if enabled.
        
        Returns:
            tuple: ((start_x, start_y, start_yaw), (goal_x, goal_y), 
                    start_region_id, goal_region_id)
            or None if sampling fails
        """
        for attempt in range(self.max_tries):
            # Sample start region and point
            start_region = random.choice(self.regions)
            start_point = self._sample_point_in_region(start_region)
            
            if start_point is None:
                self.stats["rejections"]["boundary"] += 1
                continue
            
            start_x, start_y = start_point
            
            if self.use_shaped_distance:
                # SHAPED DISTANCE SAMPLING
                # Sample target distance from bins
                target_distance = self._sample_distance_from_bins()
                
                # Sample goal at that distance
                goal_result = self._sample_goal_at_distance(
                    start_x, start_y, start_region["id"], target_distance
                )
                
                if goal_result is None:
                    self.stats["rejections"]["shaped_distance"] += 1
                    continue
                    
                goal_point, goal_region_id = goal_result
                goal_x, goal_y = goal_point
                
                # Find the goal region object
                goal_region = next(r for r in self.regions if r["id"] == goal_region_id)
                
            else:
                # ORIGINAL METHOD: Sample goal region independently
                goal_candidates = [r for r in self.regions if r["id"] != start_region["id"]]
                goal_region = random.choice(goal_candidates)
                goal_point = self._sample_point_in_region(goal_region)
                
                if goal_point is None:
                    self.stats["rejections"]["boundary"] += 1
                    continue
                    
                goal_x, goal_y = goal_point
            
            # Sample random yaw for start
            start_yaw = np.random.uniform(-np.pi, np.pi)
            
            # Calculate actual distance
            actual_distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            
            # Success
            start_pose = (start_x, start_y, start_yaw)
            goal_pose = (goal_x, goal_y)
            
            # Update stats
            self.stats["start_region_counts"][start_region["id"]] += 1
            self.stats["goal_region_counts"][goal_region["id"]] += 1
            self.stats["total_samples"] += 1
            self.stats["distances"].append(actual_distance)
            
            return start_pose, goal_pose, start_region["id"], goal_region["id"]
        
        # Failed after max_tries
        self.stats["rejections"]["max_tries"] += 1
        return None
    
    def get_stats(self):
        """Return sampling statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset sampling statistics."""
        self.stats = {
            "start_region_counts": {i: 0 for i in range(1, 14)},
            "goal_region_counts": {i: 0 for i in range(1, 14)},
            "total_samples": 0,
            "rejections": {"boundary": 0, "same_region": 0, "max_tries": 0, "shaped_distance": 0},
            "distances": [],
        }
    
    def print_stats(self):
        """Print formatted statistics including distance distribution."""
        print("\n" + "="*60)
        print("Rectangle Region Sampler Statistics")
        print("="*60)
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Shaped distance sampling: {'ENABLED' if self.use_shaped_distance else 'DISABLED'}")
        
        if self.use_shaped_distance:
            print(f"\nDistance bins and probabilities:")
            for i, (bin_range, prob) in enumerate(zip(self.distance_bins, self.bin_probs)):
                print(f"  Bin {i+1}: [{bin_range[0]:.1f}, {bin_range[1]:.1f}]m -> {prob*100:.1f}%")
        
        print(f"\nRejections:")
        for reason, count in self.stats["rejections"].items():
            print(f"  {reason}: {count}")
        
        # Distance statistics
        if self.stats["distances"]:
            distances = np.array(self.stats["distances"])
            print(f"\nDistance Distribution (start→goal Euclidean):")
            print(f"  Min:    {distances.min():.2f}m")
            print(f"  Mean:   {distances.mean():.2f}m")
            print(f"  Max:    {distances.max():.2f}m")
            print(f"  Median (p50): {np.percentile(distances, 50):.2f}m")
            print(f"  p90:    {np.percentile(distances, 90):.2f}m")
            print(f"  p95:    {np.percentile(distances, 95):.2f}m")
            
            # Threshold percentages
            pct_above_10 = 100.0 * np.sum(distances > 10.0) / len(distances)
            pct_above_12 = 100.0 * np.sum(distances > 12.0) / len(distances)
            pct_above_16 = 100.0 * np.sum(distances > 16.0) / len(distances)
            
            print(f"\nThreshold Analysis:")
            print(f"  >10m: {pct_above_10:5.1f}%")
            print(f"  >12m: {pct_above_12:5.1f}%")
            print(f"  >16m: {pct_above_16:5.1f}%")
        
        print(f"\nStart region distribution:")
        for region_id in sorted(self.stats["start_region_counts"].keys()):
            count = self.stats["start_region_counts"][region_id]
            if count > 0:
                pct = 100.0 * count / self.stats["total_samples"] if self.stats["total_samples"] > 0 else 0
                print(f"  R{region_id:2d}: {count:3d} ({pct:5.1f}%)")
        
        print(f"\nGoal region distribution:")
        for region_id in sorted(self.stats["goal_region_counts"].keys()):
            count = self.stats["goal_region_counts"][region_id]
            if count > 0:
                pct = 100.0 * count / self.stats["total_samples"] if self.stats["total_samples"] > 0 else 0
                print(f"  R{region_id:2d}: {count:3d} ({pct:5.1f}%)")
        print("="*60 + "\n")


def validate_sampler(n_samples=50, verbose=True):
    """
    Validate the rectangle region sampler.
    
    Args:
        n_samples: Number of samples to generate
        verbose: Print detailed output
        
    Returns:
        bool: True if validation passes
    """
    sampler = RectRegionSampler()
    
    if verbose:
        print(f"\nValidating RectRegionSampler with {n_samples} samples...")
        print("-" * 60)
    
    success_count = 0
    failures = []
    
    for i in range(n_samples):
        result = sampler.sample_start_goal()
        
        if result is None:
            failures.append(f"Sample {i}: Failed to generate")
            continue
        
        start_pose, goal_pose, start_region_id, goal_region_id = result
        start_x, start_y, start_yaw = start_pose
        goal_x, goal_y = goal_pose
        
        # Validate constraints
        valid = True
        
        # Check start and goal are in different regions
        if start_region_id == goal_region_id:
            failures.append(f"Sample {i}: Same region ({start_region_id})")
            valid = False
        
        # Check yaw in valid range
        if not (-np.pi <= start_yaw <= np.pi):
            failures.append(f"Sample {i}: Invalid yaw {start_yaw}")
            valid = False
        
        if valid:
            success_count += 1
    
    # Print results
    if verbose:
        sampler.print_stats()
        print(f"\nValidation Results:")
        print(f"  Success: {success_count}/{n_samples} ({100.0*success_count/n_samples:.1f}%)")
        print(f"  Failures: {len(failures)}")
        
        if failures:
            print("\nFailure details:")
            for failure in failures[:10]:  # Print first 10
                print(f"  - {failure}")
            if len(failures) > 10:
                print(f"  ... and {len(failures) - 10} more")
    
    return success_count == n_samples


def visualize_samples_in_gazebo(n_samples=50, delay_sec=2.0):
    """
    Visualize sampled poses by teleporting robot in Gazebo.
    Keeps Gazebo running, cycles through samples.
    
    Args:
        n_samples: Number of samples to visualize
        delay_sec: Delay between samples (seconds)
    """
    import rospy
    from gazebo_reset_helpers import set_robot_pose, move_goal_marker
    from geometry_msgs.msg import Twist
    
    print(f"\n{'='*60}")
    print(f"Gazebo Visualization: {n_samples} samples, {delay_sec}s delay")
    print(f"{'='*60}\n")
    
    # Initialize ROS if needed
    if not rospy.core.is_initialized():
        rospy.init_node('rect_sampler_viz', anonymous=True)
    
    # Stop robot motion
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.sleep(0.5)
    cmd_vel_pub.publish(Twist())  # Zero velocity
    
    sampler = RectRegionSampler()
    
    for i in range(n_samples):
        result = sampler.sample_start_goal()
        
        if result is None:
            print(f"[{i+1}/{n_samples}] ❌ Failed to sample")
            continue
        
        start_pose, goal_pose, start_region_id, goal_region_id = result
        start_x, start_y, start_yaw = start_pose
        goal_x, goal_y = goal_pose
        
        print(f"[{i+1}/{n_samples}] Start: R{start_region_id:2d} ({start_x:6.2f}, {start_y:6.2f}, {start_yaw:5.2f}) | "
              f"Goal: R{goal_region_id:2d} ({goal_x:6.2f}, {goal_y:6.2f})")
        
        # Stop robot
        cmd_vel_pub.publish(Twist())
        
        # Teleport robot to start
        success = set_robot_pose('turtlebot3_burger', start_x, start_y, start_yaw, z=0.07)
        if not success:
            print(f"  ⚠ Failed to set robot pose")
        
        # Move goal marker
        move_goal_marker('target', goal_x, goal_y, z=0.01)
        
        # Wait
        rospy.sleep(delay_sec)
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    sampler.print_stats()


if __name__ == "__main__":
    import sys
    
    # Check for visualization mode
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        delay_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        visualize_samples_in_gazebo(n_samples=n_samples, delay_sec=delay_sec)
    else:
        # Run validation
        success = validate_sampler(n_samples=100, verbose=True)
        
        if success:
            print("\n✅ Validation PASSED: All samples valid!")
        else:
            print("\n❌ Validation FAILED: Some samples invalid!")
            exit(1)
