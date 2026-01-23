# Spawn and Goal Sampling Module
import numpy as np

# Small House World - Safe open-space poses (x, y, yaw)
SMALL_HOUSE_START_POSES = [
    (-3.5, 1.0, 0.0), (-3.0, 0.5, 1.57), (-2.5, 1.5, -1.57), (-3.0, 2.0, 0.0),
    (-1.0, 0.0, 0.0), (-0.5, 0.5, 1.57), (0.0, 0.0, -1.57),
    (2.0, 1.5, 3.14), (2.5, 0.5, -1.57), (3.0, 1.0, 0.0),
    (1.0, -2.0, 1.57), (0.5, -2.5, 0.0), (1.5, -2.0, -1.57),
    (-1.5, 2.5, 0.0), (-2.0, 3.0, 1.57),
    (0.0, 1.0, 0.0), (-1.0, 1.5, 1.57), (1.0, 1.0, -1.57),
]

SMALL_HOUSE_GOAL_POINTS = [
    (-3.5, 0.5), (-3.0, 1.5), (-2.5, 2.0), (-3.5, 2.5), (-4.0, 1.0), (-2.0, 1.0), (-3.0, 0.0),
    (-1.0, 0.5), (-0.5, 0.0), (0.0, 0.5), (-1.5, 0.0), (0.5, 0.0), (-1.0, -0.5),
    (2.0, 0.5), (2.5, 1.0), (3.0, 1.5), (2.0, 2.0), (3.5, 1.0), (2.5, 0.0), (3.0, 0.5),
    (1.0, -2.5), (0.5, -2.0), (1.5, -2.5), (1.0, -3.0), (0.0, -2.5), (1.5, -1.5),
    (-1.5, 2.0), (-2.0, 2.5), (-1.0, 3.0), (-2.5, 2.5), (-1.5, 3.5),
    (0.0, 1.5), (-1.0, 1.0), (1.0, 0.5), (0.5, 1.5), (-0.5, 1.0), (0.0, 2.0), (1.0, 1.5),
    (-4.0, 3.0), (3.5, 2.0), (2.0, -3.0), (-2.0, -1.0),
]

class GoalSpawnSampler:
    def __init__(self, world_type="small_house", min_dist=1.5, max_dist=6.0, seed=None):
        self.world_type = world_type
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.start_poses = SMALL_HOUSE_START_POSES
        self.goal_points = SMALL_HOUSE_GOAL_POINTS
        self.rng = np.random.RandomState(seed)
    
    def sample_start_and_goal(self):
        max_attempts = 100
        tries = 0
        for _ in range(max_attempts):
            tries += 1
            start_pose = self.start_poses[self.rng.randint(len(self.start_poses))]
            goal_xy = self.goal_points[self.rng.randint(len(self.goal_points))]
            dist = np.linalg.norm([start_pose[0] - goal_xy[0], start_pose[1] - goal_xy[1]])
            if self.min_dist <= dist <= self.max_dist:
                return start_pose, goal_xy, tries
        start_pose = self.start_poses[self.rng.randint(len(self.start_poses))]
        goal_xy = self.goal_points[self.rng.randint(len(self.goal_points))]
        return start_pose, goal_xy, tries
