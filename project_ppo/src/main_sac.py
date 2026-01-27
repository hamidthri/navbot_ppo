#!/usr/bin/env python3
"""
Main training script for SAC algorithm

Integrates SAC with the existing Gazebo environment for robot navigation.
Usage:
    # Training
    python3 main_sac.py --mode train
    
    # Testing
    python3 main_sac.py --mode test --actor_model models/sac_actor_12000.pth
"""

from __future__ import absolute_import, print_function
import os
import sys
import glob
import time
import numpy as np
import torch
import rospy

from arguments import get_args
from sac import SAC
from environment_new import Env


# Environment parameters (must match environment_new.py)
STATE_DIM = 16  # 10 laser + 2 past_action + 4 goal_info
ACTION_DIM = 2  # [linear_velocity, angular_velocity]
ACTION_LINEAR_MAX = 0.25  # m/s
ACTION_ANGULAR_MAX = 1.0  # rad/s
IS_TRAINING = True

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SAC Main] Using device: {device}")


def train_sac(env, hyperparameters, model_path=None):
    """
    Train SAC agent in the Gazebo environment.
    
    Args:
        env: Gazebo environment instance
        hyperparameters (dict): Training hyperparameters
        model_path (str): Path to pre-trained model (optional)
    """
    print(f"[SAC Main] Start Training...", flush=True)
    print(f"[SAC Main] State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    print(f"[SAC Main] Device: {device}")
    
    # Create SAC agent
    agent = SAC(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=device,
        **hyperparameters
    )
    
    # Load existing model if specified
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"[SAC Main] Loaded model from {model_path}")
    else:
        print(f"[SAC Main] Training from scratch")
    
    # Training parameters
    max_timesteps = hyperparameters.get('max_timesteps', 500000)
    max_episode_steps = hyperparameters.get('max_episode_steps', 500)
    batch_size = hyperparameters.get('batch_size', 256)
    start_timesteps = hyperparameters.get('start_timesteps', 1000)
    update_after = hyperparameters.get('update_after', 1000)
    update_every = hyperparameters.get('update_every', 50)
    eval_freq = hyperparameters.get('eval_freq', 5000)
    save_freq = hyperparameters.get('save_freq', 10000)
    
    # Create save directory
    save_dir = hyperparameters.get('save_dir', 'models/sac')
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    episode_num = 0
    total_timesteps = 0
    episode_rewards = []
    episode_steps_list = []
    success_count = 0
    collision_count = 0
    
    print(f"[SAC Main] Training for {max_timesteps} timesteps")
    print(f"[SAC Main] Random exploration for first {start_timesteps} steps")
    
    while total_timesteps < max_timesteps:
        episode_num += 1
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Reset environment
        state = env.reset()
        past_action = np.array([0.0, 0.0])
        
        episode_start_time = time.time()
        
        while not done and episode_steps < max_episode_steps:
            # Select action
            if total_timesteps < start_timesteps:
                # Random exploration
                action = np.random.uniform(
                    low=[-ACTION_LINEAR_MAX, -ACTION_ANGULAR_MAX],
                    high=[ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX],
                    size=ACTION_DIM
                )
            else:
                # SAC policy
                action = agent.select_action(state, evaluate=False)
            
            # Clip action to valid range
            action = np.clip(
                action,
                [-ACTION_LINEAR_MAX, -ACTION_ANGULAR_MAX],
                [ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX]
            )
            
            # Take action in environment
            next_state, reward, done, arrive = env.step(action, past_action)
            
            # Store transition in replay buffer
            agent.memory.add(state, action, reward, next_state, float(done))
            
            # Update state and past action
            state = next_state
            past_action = action
            
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1
            
            # Update agent
            if total_timesteps >= update_after and total_timesteps % update_every == 0:
                for _ in range(update_every):
                    stats = agent.update()
                
                # Log training stats periodically
                if total_timesteps % (update_every * 10) == 0 and stats:
                    print(f"[{total_timesteps}/{max_timesteps}] "
                          f"Critic: {stats['critic_loss']:.3f}, "
                          f"Actor: {stats['actor_loss']:.3f}, "
                          f"Alpha: {stats['alpha']:.4f}, "
                          f"Q: {stats['q1_mean']:.2f}, "
                          f"Entropy: {stats['entropy']:.3f}")
            
            # Save model periodically
            if total_timesteps % save_freq == 0:
                save_path = os.path.join(save_dir, f'sac_model_{total_timesteps}.pth')
                agent.save(save_path)
            
            # Check termination
            if arrive:
                success_count += 1
                done = True
            elif done:  # Collision
                collision_count += 1
        
        # Episode finished
        episode_time = time.time() - episode_start_time
        episode_rewards.append(episode_reward)
        episode_steps_list.append(episode_steps)
        
        # Calculate statistics
        avg_reward_100 = np.mean(episode_rewards[-100:]) if episode_rewards else 0
        success_rate_100 = success_count / min(episode_num, 100)
        collision_rate_100 = collision_count / min(episode_num, 100)
        
        # Log episode info
        status = "SUCCESS" if arrive else "COLLISION" if done else "TIMEOUT"
        print(f"\n[Episode {episode_num}] {status}")
        print(f"  Steps: {episode_steps}, Reward: {episode_reward:.2f}, Time: {episode_time:.1f}s")
        print(f"  Total timesteps: {total_timesteps}/{max_timesteps}")
        print(f"  Avg reward (100 ep): {avg_reward_100:.2f}")
        print(f"  Success rate (100 ep): {success_rate_100:.2%}")
        print(f"  Collision rate (100 ep): {collision_rate_100:.2%}")
        print(f"  Buffer size: {len(agent.memory)}")
        
        # Reset counters for success/collision rate calculation
        if episode_num % 100 == 0:
            success_count = 0
            collision_count = 0
    
    # Save final model
    final_path = os.path.join(save_dir, f'sac_model_final_{total_timesteps}.pth')
    agent.save(final_path)
    print(f"\n[SAC Main] Training completed!")
    print(f"[SAC Main] Final model saved to: {final_path}")


def test_sac(env, model_path, num_episodes=10):
    """
    Test trained SAC agent.
    
    Args:
        env: Gazebo environment instance
        model_path (str): Path to trained model
        num_episodes (int): Number of episodes to test
    """
    print(f"[SAC Main] Testing model: {model_path}", flush=True)
    
    if not os.path.exists(model_path):
        print(f"[SAC Main] Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Create SAC agent
    agent = SAC(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=device,
        automatic_entropy_tuning=False  # Not needed for evaluation
    )
    
    # Load trained model
    agent.load(model_path)
    print(f"[SAC Main] Model loaded successfully")
    
    # Test loop
    success_count = 0
    collision_count = 0
    episode_rewards = []
    episode_steps_list = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        past_action = np.array([0.0, 0.0])
        
        episode_reward = 0
        episode_steps = 0
        done = False
        max_steps = 500
        
        while not done and episode_steps < max_steps:
            # Select deterministic action (mean of policy)
            action = agent.select_action(state, evaluate=True)
            
            # Clip action
            action = np.clip(
                action,
                [-ACTION_LINEAR_MAX, -ACTION_ANGULAR_MAX],
                [ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX]
            )
            
            # Take action
            state, reward, done, arrive = env.step(action, past_action)
            past_action = action
            
            episode_reward += reward
            episode_steps += 1
            
            if arrive:
                success_count += 1
                done = True
            elif done:
                collision_count += 1
        
        episode_rewards.append(episode_reward)
        episode_steps_list.append(episode_steps)
        
        status = "SUCCESS" if arrive else "COLLISION" if done else "TIMEOUT"
        print(f"[Episode {episode}/{num_episodes}] {status} - "
              f"Steps: {episode_steps}, Reward: {episode_reward:.2f}")
    
    # Print summary
    print(f"\n[SAC Main] Test Summary:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Success rate: {success_count/num_episodes:.2%}")
    print(f"  Collision rate: {collision_count/num_episodes:.2%}")
    print(f"  Avg reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Avg steps: {np.mean(episode_steps_list):.1f} ± {np.std(episode_steps_list):.1f}")


def main(args):
    """Main entry point."""
    # Initialize ROS node
    rospy.init_node('sac_stage_1')
    
    # Create environment
    env = Env(IS_TRAINING)
    
    print(f"[SAC Main] Environment initialized")
    print(f"  State dimensions: {STATE_DIM}")
    print(f"  Action dimensions: {ACTION_DIM}")
    print(f"  Action max: {ACTION_LINEAR_MAX} m/s and {ACTION_ANGULAR_MAX} rad/s")
    
    # SAC hyperparameters
    hyperparameters = {
        # Network architecture
        'hidden_dim': 256,
        
        # Learning rates
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'lr_alpha': 3e-4,
        
        # SAC parameters
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'automatic_entropy_tuning': True,
        
        # Replay buffer
        'buffer_size': int(1e6),
        'batch_size': 256,
        
        # Training parameters
        'max_timesteps': 500000,
        'max_episode_steps': 500,
        'start_timesteps': 1000,  # Random exploration steps
        'update_after': 1000,  # Start training after this many steps
        'update_every': 50,  # Update every N steps
        'eval_freq': 5000,
        'save_freq': 10000,
        
        # Paths
        'save_dir': 'models/sac_navigation'
    }
    
    # Train or test
    if args.mode == 'train':
        train_sac(env=env, hyperparameters=hyperparameters, model_path=args.actor_model)
    else:
        if not args.actor_model:
            print(f"[SAC Main] Error: Please specify --actor_model for testing")
            sys.exit(1)
        test_sac(env=env, model_path=args.actor_model, num_episodes=10)


if __name__ == '__main__':
    args = get_args()
    main(args)
