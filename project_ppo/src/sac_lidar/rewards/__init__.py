"""
Reward functions for SAC Lidar Navigation Training

Available reward functions:
- legacy: Original distance-based reward
- lyapunov: CLF+CBF Lyapunov-based reward with safety barrier
"""

from .legacy_reward import LegacyReward
from .lyapunov_reward import LyapunovReward

REWARD_FUNCTIONS = {
    'legacy': LegacyReward,
    'lyapunov': LyapunovReward,
}

def get_reward_function(reward_type: str, **kwargs):
    """
    Factory function to get reward function by name.
    
    Args:
        reward_type: Name of reward function ('legacy', 'lyapunov')
        **kwargs: Additional arguments passed to reward function constructor
        
    Returns:
        Reward function instance
    """
    if reward_type not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward type: {reward_type}. "
                        f"Available: {list(REWARD_FUNCTIONS.keys())}")
    return REWARD_FUNCTIONS[reward_type](**kwargs)
