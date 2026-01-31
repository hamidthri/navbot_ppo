#!/usr/bin/env python3
"""
LiDAR-only SAC Training Script with Config File and Clean Logging
"""
from __future__ import absolute_import, print_function
import os
import argparse
import yaml
import numpy as np
import torch
import rospy
from torch.utils.tensorboard import SummaryWriter

from sac import SAC
from environment_small_house import Env


class ActionSpace:
    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)


STATE_DIM = 16
ACTION_DIM = 2
ACTION_LINEAR_MAX = 1.0
ACTION_ANGULAR_MAX = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_dir):
    """Save config to training directory"""
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to: {config_path}")


def train_lidar_sac(config):
    """Train LiDAR-only SAC agent"""
    
    rospy.init_node('sac_lidar_small_house')
    
    # Create save directory
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config to training directory
    save_config(config, save_dir)
    
    # Create TensorBoard writer
    log_dir = os.path.join(save_dir, 'logs')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Get reward type from config (default to 'legacy')
    reward_type = config.get('reward_type', 'legacy')
    
    # Create environment with reward type
    env = Env(is_training=True, reward_type=reward_type)
    
    print(f"[Init] Device: {device}")
    print(f"[Init] State: {STATE_DIM}D (LiDAR only)")
    print(f"[Init] Reward: {reward_type}")
    print(f"[Init] Training: {config['max_timesteps']} timesteps")
    print(f"[Init] Save dir: {save_dir}")
    print(f"[Init] TensorBoard: {log_dir}\n")
    
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
        hidden_dim=config['hidden_dim'],
        lr_actor=config['lr_actor'],
        lr_critic=config['lr_critic'],
        lr_alpha=config['lr_alpha'],
        gamma=config['gamma'],
        tau=config['tau'],
        alpha=0.2,
        automatic_entropy_tuning=config['automatic_entropy_tuning'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        action_space=action_space
    )
    
    # Training parameters
    episode_num = 0
    total_timesteps = 0
    episode_rewards = []
    
    # Stats for periodic logging
    window_successes = 0
    window_collisions = 0
    window_timeouts = 0
    window_episodes = 0
    window_rewards = []
    window_steps = []  # Track steps per episode
    last_log_timestep = 0
    
    print("=" * 80)
    print("TRAINING STARTED")
    print("=" * 80 + "\n")
    
    # Track if we should reset environment (False after arrival for continuous navigation)
    should_reset = True
    
    while total_timesteps < config['max_timesteps']:
        episode_num += 1
        episode_reward = 0
        episode_steps = 0
        episode_success = False
        episode_collision = False
        done = False
        
        # Only reset if needed (skip after arrival for continuous navigation)
        if should_reset:
            state = env.reset()
        # else: continue from current position with new goal (already spawned in setReward)
        
        past_action = np.array([0.0, 0.0])
        
        while not done and episode_steps < config['max_episode_steps']:
            # Select action
            if total_timesteps < config['start_timesteps']:
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
            
            # Track episode outcome
            if arrive:
                episode_success = True
            if done and not arrive:
                episode_collision = True
            
            state = next_state
            past_action = action
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1
            
            # Update agent
            if total_timesteps >= config['update_after'] and total_timesteps % config['update_every'] == 0:
                for _ in range(config['update_every']):
                    stats = agent.update()
                
                # Log to TensorBoard
                if stats:
                    writer.add_scalar('train/critic_loss', stats['critic_loss'], total_timesteps)
                    writer.add_scalar('train/actor_loss', stats['actor_loss'], total_timesteps)
                    writer.add_scalar('train/alpha_loss', stats['alpha_loss'], total_timesteps)
                    writer.add_scalar('train/alpha', stats['alpha'], total_timesteps)
                    writer.add_scalar('train/q1_mean', stats['q1_mean'], total_timesteps)
                    writer.add_scalar('train/q2_mean', stats['q2_mean'], total_timesteps)
                    writer.add_scalar('train/entropy', stats['entropy'], total_timesteps)
            
            # Save model
            if total_timesteps % config['save_freq'] == 0:
                save_path = os.path.join(save_dir, f'sac_{total_timesteps}.pth')
                agent.save(save_path)
        
        # Episode ended - update stats
        episode_rewards.append(episode_reward)
        window_rewards.append(episode_reward)
        window_steps.append(episode_steps)
        window_episodes += 1
        
        # Count outcome ONCE per episode (not per arrival)
        if episode_success:
            window_successes += 1
            should_reset = False  # Continue from current position for next episode
        elif episode_collision:
            window_collisions += 1
            should_reset = True  # Reset environment after collision
        else:
            window_timeouts += 1
            should_reset = True  # Reset environment after timeout
        
        # Minimal per-episode output (now includes steps)
        status = "✓" if episode_success else ("✗" if episode_collision else "⊙")
        print(f"{status} Ep{episode_num} | T:{total_timesteps} | Steps:{episode_steps} | R:{episode_reward:.1f}")
        
        # Log to TensorBoard
        writer.add_scalar('episode/reward', episode_reward, episode_num)
        writer.add_scalar('episode/steps', episode_steps, episode_num)
        writer.add_scalar('episode/buffer_size', len(agent.memory), episode_num)
        
        # Detailed logs every N timesteps
        if total_timesteps - last_log_timestep >= config['log_freq']:
            avg_reward = np.mean(window_rewards) if window_rewards else 0.0
            avg_steps = np.mean(window_steps) if window_steps else 0.0
            success_rate = window_successes / max(1, window_episodes)
            
            print("\n" + "=" * 80)
            print(f"LOGS @ {total_timesteps} timesteps")
            print("=" * 80)
            print(f"Episodes:      {window_episodes}")
            print(f"Avg Return:    {avg_reward:.2f}")
            print(f"Avg Steps:     {avg_steps:.1f}")
            print(f"Success Rate:  {success_rate:.2%} ({window_successes}/{window_episodes})")
            print(f"Collisions:    {window_collisions}")
            print(f"Timeouts:      {window_timeouts}")
            print(f"Buffer Size:   {len(agent.memory)}")
            if stats:
                print(f"Critic Loss:   {stats['critic_loss']:.4f}")
                print(f"Actor Loss:    {stats['actor_loss']:.4f}")
                print(f"Alpha:         {stats['alpha']:.4f}")
                print(f"Q1 Mean:       {stats['q1_mean']:.2f}")
                print(f"Entropy:       {stats['entropy']:.4f}")
            print("=" * 80 + "\n")
            
            # Reset window stats
            window_successes = 0
            window_collisions = 0
            window_timeouts = 0
            window_episodes = 0
            window_rewards = []
            window_steps = []
            last_log_timestep = total_timesteps
    
    # Save final model
    final_path = os.path.join(save_dir, f'sac_final_{total_timesteps}.pth')
    agent.save(final_path)
    writer.close()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final model: {final_path}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LiDAR SAC')
    parser.add_argument('--config', type=str, default='config_lidar.yaml',
                        help='Path to config file')
    
    # Allow overriding config values
    parser.add_argument('--max_timesteps', type=int, default=None,
                        help='Override max timesteps')
    parser.add_argument('--reward_type', type=str, default=None,
                        choices=['legacy', 'lyapunov'],
                        help='Override reward type (legacy or lyapunov)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom run name for save directory')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.max_timesteps is not None:
        config['max_timesteps'] = args.max_timesteps
    
    if args.reward_type is not None:
        config['reward_type'] = args.reward_type
    
    # Build save directory with run name
    reward_type = config.get('reward_type', 'legacy')
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{reward_type}_{config['max_timesteps']//1000}k"
    
    config['save_dir'] = os.path.join('models/sac_lidar', run_name)
    
    # Train
    train_lidar_sac(config)