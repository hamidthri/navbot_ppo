#!/usr/bin/env python3
"""
SAC Training Script for Robot Navigation
Simplified version without tests or extra documentation
"""

from __future__ import absolute_import, print_function
import os
import time
import numpy as np
import torch
import rospy

from sac import SAC
from environment_small_house import Env  # Small house environment


# Simple action space class for SAC
class ActionSpace:
    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)


STATE_DIM = 16
ACTION_DIM = 2
# NOTE: Environment divides linear velocity by 4 before publishing!
# So we use 4x larger bound (1.0) to get effective range [0, 0.25] m/s
ACTION_LINEAR_MAX = 1.0  # Will become 0.25 m/s after /4 in environment
ACTION_ANGULAR_MAX = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    """Train SAC agent."""
    rospy.init_node('sac_stage_1_small_house_10k')
    env = Env(is_training=True)  # Use small house environment
    
    print(f"[SAC] Device: {device}")
    print(f"[SAC] State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    print(f"[SAC] Action bounds: linear [0, {ACTION_LINEAR_MAX}], angular [{-ACTION_ANGULAR_MAX}, {ACTION_ANGULAR_MAX}]")
    print(f"[SAC] Environment: Small House (10k timesteps)")
    print(f"[SAC] Curriculum: 0.1m increment every 500 successes")
    
    # Define action space (linear [0, 1], angular [-1, 1])
    action_space = ActionSpace(
        low=[0.0, -ACTION_ANGULAR_MAX],
        high=[ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX]
    )
    
    # Create agent
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
        buffer_size=int(1e6),
        batch_size=256,
        action_space=action_space
    )
    
    # Training parameters
    max_timesteps = 10000  # Changed from 500000 to 10000 for test
    max_episode_steps = 500
    start_timesteps = 1000
    update_after = 1000
    update_every = 50
    save_freq = 5000  # Save at 5k and 10k
    save_dir = 'models/sac_small_house_10k'
    os.makedirs(save_dir, exist_ok=True)
    
    episode_num = 0
    total_timesteps = 0
    episode_rewards = []
    success_count = 0
    collision_count = 0
    
    print(f"[SAC] Training for {max_timesteps} timesteps")
    print(f"[SAC] Saving models to: {save_dir}")
    
    while total_timesteps < max_timesteps:
        episode_num += 1
        episode_reward = 0
        episode_steps = 0
        episode_successes = 0  # track successes inside this episode
        done = False
        
        state = env.reset()
        past_action = np.array([0.0, 0.0])
        
        while not done and episode_steps < max_episode_steps:
            # Select action
            if total_timesteps < start_timesteps:
                # Random exploration: linear [0, 1], angular [-1, 1]
                action = np.array([
                    np.random.uniform(0.0, ACTION_LINEAR_MAX),
                    np.random.uniform(-ACTION_ANGULAR_MAX, ACTION_ANGULAR_MAX)
                ])
            else:
                action = agent.select_action(state, evaluate=False)
            
            # Clip actions: linear [0, 1], angular [-1, 1] (same as PPO)
            action[0] = np.clip(action[0], 0.0, ACTION_LINEAR_MAX)
            action[1] = np.clip(action[1], -ACTION_ANGULAR_MAX, ACTION_ANGULAR_MAX)
            
            next_state, reward, done, arrive = env.step(action, past_action)
            agent.memory.add(state, action, reward, next_state, float(done))
            
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
                          f"Q: {stats['q1_mean']:.2f}")
            
            if total_timesteps % save_freq == 0:
                save_path = os.path.join(save_dir, f'sac_{total_timesteps}.pth')
                agent.save(save_path)
            
            # Track successes during the episode (environment continues after success)
            if arrive:
                success_count += 1
                episode_successes += 1
            # Track collisions which end the episode
            if done:
                collision_count += 1
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        success_rate = success_count / max(1, episode_num)  # successes per episode (avg)
        
        status = f"SUCCESS x{episode_successes}" if episode_successes > 0 else ("COLLISION" if done else "TIMEOUT")
        print(f"[Ep {episode_num}] {status} | Steps: {episode_steps} | "
              f"Reward: {episode_reward:.1f} | Avg: {avg_reward:.1f} | "
              f"Total Success: {success_count} ({success_rate:.2f}/ep) | Total: {total_timesteps}")
        
        if episode_num % 100 == 0:
            success_count = 0
            collision_count = 0
    
    final_path = os.path.join(save_dir, f'sac_final_{total_timesteps}.pth')
    agent.save(final_path)
    print(f"[SAC] Training complete! Model: {final_path}")


if __name__ == '__main__':
    train()
