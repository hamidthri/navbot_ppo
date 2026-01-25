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
    """
    
    def __init__(self, robot_radius=0.105, clearance_margin=0.30, max_tries=100):
        """
        Args:
            robot_radius: Robot radius in meters
            clearance_margin: Safety margin from rectangle boundaries
            max_tries: Maximum attempts before giving up
        """
        self.robot_radius = robot_radius
        self.clearance_margin = clearance_margin
        self.total_clearance = robot_radius + clearance_margin
        self.max_tries = max_tries
        self.regions = RECT_REGIONS
        
        # Statistics
        self.stats = {
            "start_region_counts": {i: 0 for i in range(1, 14)},
            "goal_region_counts": {i: 0 for i in range(1, 14)},
            "total_samples": 0,
            "rejections": {"boundary": 0, "same_region": 0, "max_tries": 0},
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
    
    def sample_start_goal(self):
        """
        Sample start pose and goal position from different regions.
        
        Returns:
            tuple: ((start_x, start_y, start_yaw), (goal_x, goal_y), 
                    start_region_id, goal_region_id)
            or None if sampling fails
        """
        for attempt in range(self.max_tries):
            # Sample two different regions
            start_region = random.choice(self.regions)
            
            # Filter out start_region for goal sampling
            goal_candidates = [r for r in self.regions if r["id"] != start_region["id"]]
            goal_region = random.choice(goal_candidates)
            
            # Sample points
            start_point = self._sample_point_in_region(start_region)
            goal_point = self._sample_point_in_region(goal_region)
            
            if start_point is None or goal_point is None:
                self.stats["rejections"]["boundary"] += 1
                continue
            
            # Sample random yaw for start
            start_yaw = np.random.uniform(-np.pi, np.pi)
            
            # Success
            start_pose = (start_point[0], start_point[1], start_yaw)
            goal_pose = goal_point
            
            # Update stats
            self.stats["start_region_counts"][start_region["id"]] += 1
            self.stats["goal_region_counts"][goal_region["id"]] += 1
            self.stats["total_samples"] += 1
            
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
            "rejections": {"boundary": 0, "same_region": 0, "max_tries": 0},
        }
    
    def print_stats(self):
        """Print formatted statistics."""
        print("\n" + "="*60)
        print("Rectangle Region Sampler Statistics")
        print("="*60)
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"\nRejections:")
        for reason, count in self.stats["rejections"].items():
            print(f"  {reason}: {count}")
        
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
