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

STAGE1_START_POSES = [
    (0.0, 0.0, 0.0), (0.5, 0.5, 0.785), (-0.5, 0.5, 2.356), (0.5, -0.5, -0.785),
    (-0.5, -0.5, -2.356), (1.0, 0.0, 0.0), (0.0, 1.0, 1.57), (-1.0, 0.0, 3.14),
    (0.0, -1.0, -1.57), (1.0, 1.0, 0.785),
]

STAGE1_GOAL_POINTS = [
    (3.0, 3.0), (3.5, 2.5), (2.5, 3.5), (4.0, 3.0), (-3.0, 3.0), (-3.5, 2.5), (-2.5, 3.5), (-4.0, 3.0),
    (3.0, -3.0), (3.5, -2.5), (2.5, -3.5), (4.0, -3.0), (-3.0, -3.0), (-3.5, -2.5), (-2.5, -3.5), (-4.0, -3.0),
    (4.0, 0.0), (-4.0, 0.0), (0.0, 4.0), (0.0, -4.0), (3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0),
    (2.0, 2.0), (-2.0, 2.0), (2.0, -2.0), (-2.0, -2.0),
]

class GoalSpawnSampler:
    def __init__(self, world_type='small_house', min_dist=1.5, max_dist=6.0, seed=None):
        self.world_type = world_type
        self.min_dist = min_dist
        self.max_dist = max_dist
        if world_type == 'small_house':
            self.start_poses = SMALL_HOUSE_START_POSES
            self.goal_points = SMALL_HOUSE_GOAL_POINTS
        elif world_type == 'stage1':
            self.start_poses = STAGE1_START_POSES
            self.goal_points = STAGE1_GOAL_POINTS
        else:
            raise ValueError(f'Unknown world_type: {world_type}')
        self.rng = np.random.RandomState(seed)
    
    def sample_start_and_goal(self):
        max_attempts = 100
        for _ in range(max_attempts):
            start_pose = self.start_poses[self.rng.randint(len(self.start_poses))]
            goal_xy = self.goal_points[self.rng.randint(len(self.goal_points))]
            dist = np.linalg.norm([start_pose[0] - goal_xy[0], start_pose[1] - goal_xy[1]])
            if self.min_dist <= dist <= self.max_dist:
                return start_pose, goal_xy
        start_pose = self.start_poses[self.rng.randint(len(self.start_poses))]
        goal_xy = self.goal_points[self.rng.randint(len(self.goal_points))]
        return start_pose, goal_xy
    
    def validate_open_space(self, scan_msg, open_thresh=0.4):
        if scan_msg is None:
            return False
        ranges = np.array(scan_msg.ranges)
        valid_ranges = ranges[(ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)]
        if len(valid_ranges) == 0:
            return False
        min_range = np.min(valid_ranges)
        return min_range > open_thresh

if __name__ == '__main__':
    print('Testing Small House Sampler:')
    sampler = GoalSpawnSampler(world_type='small_house', seed=42)
    print('5 sampled (start, goal) pairs:')
    for i in range(5):
        start, goal = sampler.sample_start_and_goal()
        dist = np.linalg.norm([start[0] - goal[0], start[1] - goal[1]])
        print(f'  {i+1}. Start: ({start[0]:5.2f}, {start[1]:5.2f}, {start[2]:5.2f}) -> Goal: ({goal[0]:5.2f}, {goal[1]:5.2f})  [dist={dist:.2f}m]')
