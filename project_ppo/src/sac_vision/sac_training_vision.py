#!/usr/bin/env python3
"""
Vision SAC Training Script with Reward Normalization
FIXED: Normalizes rewards before storing in replay buffer
"""
import os
import argparse
import yaml
import numpy as np
import torch
import rospy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from geometry_msgs.msg import Twist

from sac_vision import VisionSAC
from vision_backbones import get_vision_backbone
from environment_small_house import Env


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


class ActionSpace:
    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)


def train_vision_sac(config):
    """Train vision SAC with proper reward normalization"""
    
    rospy.init_node('sac_vision_training')
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    save_dir = os.path.join(config['save_dir'], f"{config['fusion_type']}_{config['backbone']}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    save_config(config, save_dir)
    
    # Setup logging
    log_dir = os.path.join(save_dir, 'logs')
    writer = SummaryWriter(log_dir=log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'training_{timestamp}.log')
    
    # Create environment
    reward_type = config.get('reward_type', 'legacy')
    env = Env(is_training=True, use_vision=True, reward_type=reward_type)
    
    # Print initialization info
    print(f"[Init] Device: {device}")
    print(f"[Init] Backbone: {config['backbone']}, Fusion: {config['fusion_type']}")
    print(f"[Init] Lidar: 16D → {config['lidar_encoder_dim']}D")
    print(f"[Init] Reward type: {reward_type}")
    print(f"[Init] Training: {config['max_timesteps']} timesteps")
    print(f"[Init] Save dir: {save_dir}")
    print(f"[Init] TensorBoard: {log_dir}")
    print(f"[Init] Reward normalization: scale by /{config['reward_scale']}")
    print()
    
    # Create vision backbone
    vision_backbone = get_vision_backbone(
        architecture=config['backbone'],
        pretrained=True,
        output_dim=512
    )
    vision_backbone.eval()
    vision_backbone.to(device)
    
    # Action space
    action_space = ActionSpace(low=[0.0, -1.0], high=[1.0, 1.0])
    
    # Create agent
    agent = VisionSAC(
        lidar_dim=16,
        action_dim=2,
        vision_backbone=vision_backbone,
        device=device,
        hidden_dim=config['hidden_dim'],
        lr_actor=config['lr_actor'],
        lr_critic=config['lr_critic'],
        lr_alpha=config['lr_alpha'],
        gamma=config['gamma'],
        tau=config['tau'],
        automatic_entropy_tuning=True,
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        action_space=action_space,
        fusion_type=config['fusion_type'],
        lidar_encoder_dim=config['lidar_encoder_dim'],
        # Stability parameters
        gradient_clip_norm=config['gradient_clip_norm'],
        reward_scale=config['reward_scale'],
        use_reward_normalization=config.get('use_reward_normalization', False),
        q_value_clip=(config['q_value_clip_min'], config['q_value_clip_max'])
    )
    
    # Resume from checkpoint if specified
    start_timesteps = 0
    if config.get('resume_checkpoint'):
        checkpoint_path = config['resume_checkpoint']
        if os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
            # Extract timestep from checkpoint name (e.g., sac_70000.pth -> 70000)
            try:
                checkpoint_name = os.path.basename(checkpoint_path)
                start_timesteps = int(checkpoint_name.replace('sac_', '').replace('.pth', '').replace('final_', ''))
                print(f"[Resume] Loaded checkpoint from {checkpoint_path}")
                print(f"[Resume] Starting from timestep {start_timesteps}")
            except ValueError:
                print(f"[Resume] Loaded checkpoint but couldn't parse timestep from {checkpoint_path}")
                print(f"[Resume] Starting from timestep 0")
        else:
            print(f"[Warning] Checkpoint not found: {checkpoint_path}")
    
    # Training loop
    episode_num = 0
    total_timesteps = start_timesteps
    
    # Stats for periodic logging
    window_successes = 0
    window_collisions = 0
    window_timeouts = 0
    window_episodes = 0
    window_rewards = []
    window_steps = []  # Track steps per episode
    last_log_timestep = 0
    
    # Track raw rewards for monitoring
    raw_reward_stats = {'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0}
    
    print("=" * 80)
    print("TRAINING STARTED")
    print("=" * 80 + "\n")
    
    # Track if we should reset environment (False after arrival for continuous navigation)
    should_reset = True
    stats = None  # Initialize stats variable for logging
    
    while total_timesteps < config['max_timesteps']:
        episode_num += 1
        episode_reward = 0
        episode_raw_reward = 0  # Track raw reward
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
                    np.random.uniform(0.0, 1.0),
                    np.random.uniform(-1.0, 1.0)
                ])
            else:
                action = agent.select_action(state, evaluate=False)
            
            # Clip actions
            action[0] = np.clip(action[0], 0.0, 1.0)
            action[1] = np.clip(action[1], -1.0, 1.0)
            
            # Environment step
            next_state, raw_reward, done, arrive = env.step(action, past_action)
            
            # NORMALIZE REWARD BEFORE STORING
            normalized_reward = agent.normalize_reward(raw_reward)
            
            # Store normalized reward in replay buffer
            agent.memory.add(state, action, normalized_reward, next_state, float(done))
            
            # Track statistics
            raw_reward_stats['min'] = min(raw_reward_stats['min'], raw_reward)
            raw_reward_stats['max'] = max(raw_reward_stats['max'], raw_reward)
            raw_reward_stats['sum'] += raw_reward
            raw_reward_stats['count'] += 1
            
            if arrive:
                episode_success = True
            if done and not arrive:
                episode_collision = True
            
            state = next_state
            past_action = action
            episode_reward += normalized_reward  # Track normalized reward
            episode_raw_reward += raw_reward     # Track raw reward too
            episode_steps += 1
            total_timesteps += 1
            
            # Update agent
            if total_timesteps >= config['update_after'] and total_timesteps % config['update_every'] == 0:
                # CRITICAL: Stop robot during backpropagation to prevent random movements
                # Vision processing takes time, so we need to ensure robot doesn't continue
                # with last action during gradient computation
                stop_cmd = Twist()
                env.pub_cmd_vel.publish(stop_cmd)
                
                # REDUCE number of gradient steps to prevent robot freezing
                gradient_steps = min(config['update_every'], 20)  # Max 20 steps per update
                
                for _ in range(gradient_steps):
                    stats = agent.update()
                
                # Keep robot stopped after updates - next env.step() will send new action
                env.pub_cmd_vel.publish(stop_cmd)
                
                # Log to TensorBoard
                if stats:
                    writer.add_scalar('train/critic_loss', stats['critic_loss'], total_timesteps)
                    writer.add_scalar('train/actor_loss', stats['actor_loss'], total_timesteps)
                    writer.add_scalar('train/alpha_loss', stats['alpha_loss'], total_timesteps)
                    writer.add_scalar('train/alpha', stats['alpha'], total_timesteps)
                    writer.add_scalar('train/q1_mean', stats['q1_mean'], total_timesteps)
                    writer.add_scalar('train/q2_mean', stats['q2_mean'], total_timesteps)
            
            # Save model
            if total_timesteps % config['save_freq'] == 0:
                save_path = os.path.join(save_dir, f'sac_{total_timesteps}.pth')
                agent.save(save_path)
        
        # Episode ended
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
        print(f"{status} Ep{episode_num} | T:{total_timesteps} | Steps:{episode_steps} | R:{episode_reward:.1f} (raw:{episode_raw_reward:.1f})")
        
        # Log to TensorBoard
        writer.add_scalar('episode/reward_normalized', episode_reward, episode_num)
        writer.add_scalar('episode/reward_raw', episode_raw_reward, episode_num)
        writer.add_scalar('episode/steps', episode_steps, episode_num)
        writer.add_scalar('episode/buffer_size', len(agent.memory), episode_num)
        
        # Detailed logs every N timesteps
        if total_timesteps - last_log_timestep >= config['log_freq']:
            avg_reward = np.mean(window_rewards) if window_rewards else 0.0
            avg_steps = np.mean(window_steps) if window_steps else 0.0
            success_rate = window_successes / max(1, window_episodes)
            
            avg_raw_reward = raw_reward_stats['sum'] / max(1, raw_reward_stats['count'])
            
            print("\n" + "=" * 80)
            print(f"LOGS @ {total_timesteps} timesteps")
            print("=" * 80)
            print(f"Episodes:      {window_episodes}")
            print(f"Avg Return (normalized): {avg_reward:.2f}")
            print(f"Avg Return (raw):        {avg_raw_reward:.2f}")
            print(f"Avg Steps:     {avg_steps:.1f}")
            print(f"Raw Reward Range:        [{raw_reward_stats['min']:.1f}, {raw_reward_stats['max']:.1f}]")
            print(f"Success Rate:  {success_rate:.2%} ({window_successes}/{window_episodes})")
            print(f"Collisions:    {window_collisions}")
            print(f"Timeouts:      {window_timeouts}")
            print(f"Buffer Size:   {len(agent.memory)}")
            if stats:
                print(f"Critic Loss:   {stats['critic_loss']:.4f}")
                print(f"Actor Loss:    {stats['actor_loss']:.4f}")
                print(f"Alpha:         {stats['alpha']:.4f}")
                print(f"Q1 Mean:       {stats['q1_mean']:.2f}")
            print("=" * 80 + "\n")
            
            # Reset window stats
            window_successes = 0
            window_collisions = 0
            window_timeouts = 0
            window_episodes = 0
            window_rewards = []
            window_steps = []
            raw_reward_stats = {'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0}
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
    parser = argparse.ArgumentParser(description='Train Vision SAC')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--fusion_type', type=str, default=None,
                        help='Override fusion type')
    parser.add_argument('--max_timesteps', type=int, default=None,
                        help='Override max timesteps')
    parser.add_argument('--reward_type', type=str, choices=['legacy', 'lyapunov'],
                        default=None, help='Reward function type: legacy or lyapunov')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for this run (subdirectory in save_dir)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., models/sac_vision/run_name/sac_70000.pth)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.fusion_type is not None:
        config['fusion_type'] = args.fusion_type
    if args.max_timesteps is not None:
        config['max_timesteps'] = args.max_timesteps
    if args.reward_type is not None:
        config['reward_type'] = args.reward_type
    if args.run_name is not None:
        # Append run_name to save_dir
        config['save_dir'] = os.path.join(config['save_dir'], args.run_name)
    
    # Resume from checkpoint
    config['resume_checkpoint'] = args.resume
    
    # Train
    train_vision_sac(config)