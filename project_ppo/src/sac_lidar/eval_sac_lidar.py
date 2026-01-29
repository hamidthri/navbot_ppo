#!/usr/bin/env python3
"""
LiDAR SAC Evaluation Script
Loads a trained model and evaluates it in the small house environment
"""
from __future__ import absolute_import, print_function
import os
import sys
import argparse
import numpy as np
import torch
import rospy

from sac import SAC
from environment_small_house import Env


# Simple action space class
class ActionSpace:
    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)


# Configuration
STATE_DIM = 16  # LiDAR only
ACTION_DIM = 2
ACTION_LINEAR_MAX = 1.0
ACTION_ANGULAR_MAX = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_lidar_sac(
    model_path,
    num_episodes=10,
    max_steps=500,
    render=False
):
    """
    Evaluate trained LiDAR SAC agent
    
    Args:
        model_path: Path to saved model checkpoint
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render (Gazebo GUI should be running)
    """
    rospy.init_node('sac_lidar_eval')
    
    print(f"\n{'='*60}")
    print(f"LiDAR SAC Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Create environment (LiDAR-only)
    env = Env(is_training=False)
    
    # Define action space
    action_space = ActionSpace(
        low=[0.0, -ACTION_ANGULAR_MAX],
        high=[ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX]
    )
    
    # Create agent
    agent = SAC(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=device,
        action_space=action_space
    )
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    agent.load(model_path)
    print(f"✓ Model loaded from {model_path}\n")
    
    # Evaluation loop
    episode_rewards = []
    episode_steps_list = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    for ep in range(num_episodes):
        episode_reward = 0
        episode_steps = 0
        episode_successes = 0
        done = False
        
        state = env.reset()
        past_action = np.array([0.0, 0.0])
        
        while not done and episode_steps < max_steps:
            # Select action (evaluation mode - deterministic)
            action = agent.select_action(state, evaluate=True)
            
            # Clip actions
            action[0] = np.clip(action[0], 0.0, ACTION_LINEAR_MAX)
            action[1] = np.clip(action[1], -ACTION_ANGULAR_MAX, ACTION_ANGULAR_MAX)
            
            next_state, reward, done, arrive = env.step(action, past_action)
            
            if arrive:
                episode_successes += 1
            
            state = next_state
            past_action = action
            episode_reward += reward
            episode_steps += 1
        
        episode_rewards.append(episode_reward)
        episode_steps_list.append(episode_steps)
        
        if episode_successes > 0:
            success_count += 1
            status = f"✓ SUCCESS x{episode_successes}"
        elif done:
            collision_count += 1
            status = "✗ COLLISION"
        else:
            timeout_count += 1
            status = "⏱ TIMEOUT"
        
        print(f"[Ep {ep+1}/{num_episodes}] {status} | "
              f"Steps: {episode_steps} | Reward: {episode_reward:.1f}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes:        {num_episodes}")
    print(f"Success Rate:    {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Collision Rate:  {collision_count}/{num_episodes} ({100*collision_count/num_episodes:.1f}%)")
    print(f"Timeout Rate:    {timeout_count}/{num_episodes} ({100*timeout_count/num_episodes:.1f}%)")
    print(f"Avg Reward:      {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Avg Steps:       {np.mean(episode_steps_list):.1f} ± {np.std(episode_steps_list):.1f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate LiDAR SAC Agent')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    
    args = parser.parse_args()
    
    evaluate_lidar_sac(
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.max_steps
    )
