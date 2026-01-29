#!/usr/bin/env python3
"""
LiDAR-only SAC Training Script for Small House
10k timesteps training to verify pipeline
"""
from __future__ import absolute_import, print_function
import os
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


def train_lidar_sac(
    max_timesteps=10000,
    save_dir='models/sac_lidar_10k',
    buffer_size=int(1e6)  # Large buffer for LiDAR (no images)
):
    """
    Train LiDAR-only SAC agent
    
    Args:
        max_timesteps: Total training timesteps
        save_dir: Directory to save models
        buffer_size: Replay buffer capacity
    """
    rospy.init_node('sac_lidar_small_house')
    
    # Create environment (LiDAR-only, no vision support in this version)
    env = Env(is_training=True)
    
    print(f"[SAC-LiDAR] Device: {device}")
    print(f"[SAC-LiDAR] State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    print(f"[SAC-LiDAR] Training for {max_timesteps} timesteps")
    
    # Define action space
    action_space = ActionSpace(
        low=[0.0, -ACTION_ANGULAR_MAX],
        high=[ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX]
    )
    
    # Create SAC agent
    agent = SAC(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=device,
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        buffer_size=buffer_size,
        batch_size=256,  # Larger batch for LiDAR-only
        action_space=action_space
    )
    
    # Training parameters
    max_episode_steps = 500
    start_timesteps = 1000
    update_after = 1000
    update_every = 50
    save_freq = 5000
    os.makedirs(save_dir, exist_ok=True)
    
    episode_num = 0
    total_timesteps = 0
    episode_rewards = []
    success_count = 0
    collision_count = 0
    
    print(f"[SAC-LiDAR] Saving models to: {save_dir}")
    print(f"[SAC-LiDAR] Starting training...")
    
    while total_timesteps < max_timesteps:
        episode_num += 1
        episode_reward = 0
        episode_steps = 0
        episode_successes = 0
        done = False
        
        state = env.reset()  # Returns LiDAR array (16,)
        past_action = np.array([0.0, 0.0])
        
        while not done and episode_steps < max_episode_steps:
            # Select action
            if total_timesteps < start_timesteps:
                # Random exploration
                action = np.array([
                    np.random.uniform(0.0, ACTION_LINEAR_MAX),
                    np.random.uniform(-ACTION_ANGULAR_MAX, ACTION_ANGULAR_MAX)
                ])
            else:
                action = agent.select_action(state, evaluate=False)
            
            # Clip actions
            action[0] = np.clip(action[0], 0.0, ACTION_LINEAR_MAX)
            action[1] = np.clip(action[1], -ACTION_ANGULAR_MAX, ACTION_ANGULAR_MAX)
            
            next_state, reward, done, arrive = env.step(action, past_action)
            agent.memory.add(state, action, reward, next_state, float(done))
            
            # Track successes during episode
            if arrive:
                success_count += 1
                episode_successes += 1
            
            # Track collisions
            if done:
                collision_count += 1
            
            state = next_state
            past_action = action
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1
            
            # Update agent
            if total_timesteps >= update_after and total_timesteps % update_every == 0:
                for _ in range(update_every):
                    stats = agent.update()
                
                if total_timesteps % (update_every * 10) == 0 and stats:
                    print(f"[{total_timesteps}/{max_timesteps}] "
                          f"Critic: {stats['critic_loss']:.3f}, "
                          f"Actor: {stats['actor_loss']:.3f}, "
                          f"Alpha: {stats['alpha']:.4f}, "
                          f"Q1: {stats['q1_mean']:.2f}")
            
            if total_timesteps % save_freq == 0:
                save_path = os.path.join(save_dir, f'sac_lidar_{total_timesteps}.pth')
                agent.save(save_path)
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        success_rate = success_count / max(1, episode_num)
        
        status = f"SUCCESS x{episode_successes}" if episode_successes > 0 else ("COLLISION" if done else "TIMEOUT")
        print(f"[Ep {episode_num}] {status} | Steps: {episode_steps} | "
              f"Reward: {episode_reward:.1f} | Avg: {avg_reward:.1f} | "
              f"Total Success: {success_count} ({success_rate:.2f}/ep) | Total: {total_timesteps}")
        
        if episode_num % 100 == 0:
            success_count = 0
            collision_count = 0
    
    final_path = os.path.join(save_dir, f'sac_lidar_final_{total_timesteps}.pth')
    agent.save(final_path)
    print(f"[SAC-LiDAR] Training complete! Model: {final_path}")


if __name__ == '__main__':
    # Train LiDAR-only SAC for 10k timesteps
    train_lidar_sac(
        max_timesteps=10000,
        save_dir='models/sac_lidar_10k',
        buffer_size=int(1e6)  # 1M buffer (LiDAR is memory-efficient)
    )
