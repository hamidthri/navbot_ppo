#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import print_function
import gym
import sys
import torch

from arguments import get_args
from ppo import PPO
from net_actor import NetActor
from net_critic import NetCritic
from eval_policy import eval_policy
import rospy
import numpy as np

from environment_new import Env
import os, glob


state_dim = 16
action_dim = 2
action_linear_max = 0.25  # m/s
action_angular_max = 1  # rad/s
is_training = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(env, hyperparameters, actor_model, critic_model, max_timesteps=5000):
    """
        Trains the model.

        Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training
            max_timesteps - maximum timesteps to train for

        Return:
            None
    """
    print(f"Start Training ... ", flush=True)
    
    # Get actual state_dim from environment (may include vision features)
    actual_state_dim = hyperparameters.pop('state_dim', state_dim)

    # Create a model for PPO.
    agent = PPO(policy_class=NetActor, value_func=NetCritic, env=env, state_dim=actual_state_dim, action_dim=action_dim,
                **hyperparameters)
    past_action = np.array([0., 0.])

    # Tries to load in an existing actor/critic model to continue training on
    checkpoint_pattern = os.path.join(agent.checkpoint_dir, 'critic_step*.pth')
    critic_paths = sorted(glob.glob(checkpoint_pattern))
    actor_pattern = os.path.join(agent.checkpoint_dir, 'actor_step*.pth')
    actor_paths = sorted(glob.glob(actor_pattern))
    
    # Only load if state_dim matches (check if we have vision features)
    load_checkpoint = False
    if critic_paths != [] and actor_paths != []:
        # Try to load and check dimensions
        try:
            checkpoint = torch.load(actor_paths[-1])
            # Check first layer input dimension
            first_key = list(checkpoint.keys())[0]
            if 'bn1' in first_key or 'rb1.fc1' in first_key:
                # Extract input dim from first layer
                for key in checkpoint.keys():
                    if 'rb1.fc1.weight' in key:
                        checkpoint_input_dim = checkpoint[key].shape[1]
                        if checkpoint_input_dim == actual_state_dim:
                            load_checkpoint = True
                            print(f"Checkpoint state_dim matches ({checkpoint_input_dim}), will load", flush=True)
                        else:
                            print(f"Checkpoint state_dim mismatch: checkpoint={checkpoint_input_dim}, current={actual_state_dim}", flush=True)
                            print(f"Training from scratch with new state_dim", flush=True)
                        break
        except Exception as e:
            print(f"Could not check checkpoint dimensions: {e}", flush=True)
    
    if load_checkpoint:
        print(f"Loading in {actor_paths[-1]} and {critic_paths[-1]}...", flush=True)
        agent.actor.load_state_dict(torch.load(actor_paths[-1]))
        agent.critic.load_state_dict(torch.load(critic_paths[-1]))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '':  # Don't train from scratch if user accidentally forgets actor/critic model
        print(
            f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    agent.learn(total_timesteps=max_timesteps, past_action=past_action)
    #

def test(env, actor_model):
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = state_dim
    act_dim = action_dim

    # Build our policy the same way we build our actor model in PPO
    policy = NetActor(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True)

def evaluate(env, hyperparameters, actor_model, critic_model, num_episodes):
    """
        Evaluation mode: Load checkpoint and run N episodes to compute metrics.
    """
    import csv
    import time as time_module
    
    print(f"Evaluation Mode: Running {num_episodes} episodes", flush=True)
    
    method_name = hyperparameters.get('method_name', 'baseline')
    exp_id = hyperparameters['exp_id']
    actual_state_dim = hyperparameters.pop('state_dim', state_dim)
    
    # Load model
    if actor_model == '':
        # Try to find latest checkpoint
        actor_path = sorted(glob.glob(f'../../../models/{exp_id}/ppo_actor_{method_name}_*.pth'))
        if not actor_path:
            print("No checkpoint found for evaluation. Exiting.", flush=True)
            sys.exit(0)
        actor_model = actor_path[-1]
    
    print(f"Loading actor model: {actor_model}", flush=True)
    
    policy = NetActor(actual_state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(actor_model))
    policy.eval()
    
    # Prepare CSV
    eval_csv_path = f'../../../record/{method_name}_eval_episodes.csv'
    os.makedirs(os.path.dirname(eval_csv_path), exist_ok=True)
    
    with open(eval_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'success', 'collision', 'timeout', 'length', 'return', 'path_length'])
    
    # Run episodes
    metrics = {'success': 0, 'collision': 0, 'timeout': 0, 'lengths': [], 'returns': [], 'path_lengths': []}
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        arrive = False
        ep_return = 0
        ep_length = 0
        path_length = 0.0
        past_action = np.array([0., 0.])
        prev_pos = None
        
        while ep_length < hyperparameters['max_timesteps_per_episode']:
            # Deterministic action
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                action = policy(obs_tensor).cpu().numpy()
            
            # Get current position for path length
            curr_pos = np.array([env.position.x, env.position.y])
            if prev_pos is not None:
                path_length += np.linalg.norm(curr_pos - prev_pos)
            prev_pos = curr_pos
            
            obs, rew, done, arrive = env.step(action, past_action)
            past_action = action
            ep_return += rew
            ep_length += 1
            
            if done or arrive:
                break
        
        # Determine episode outcome
        success = 1 if arrive else 0
        collision = 1 if done and not arrive else 0
        timeout = 1 if (not done and not arrive and ep_length >= hyperparameters['max_timesteps_per_episode']) else 0
        
        metrics['success'] += success
        metrics['collision'] += collision
        metrics['timeout'] += timeout
        metrics['lengths'].append(ep_length)
        metrics['returns'].append(ep_return)
        metrics['path_lengths'].append(path_length)
        
        # Write to CSV
        with open(eval_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, success, collision, timeout, ep_length, ep_return, path_length])
        
        if (ep + 1) % 10 == 0:
            print(f"Evaluated {ep+1}/{num_episodes} episodes", flush=True)
    
    # Print summary
    print("\n" + "="*60, flush=True)
    print("EVALUATION SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"Method: {method_name}", flush=True)
    print(f"Episodes: {num_episodes}", flush=True)
    print(f"Success Rate: {metrics['success']/num_episodes*100:.2f}%", flush=True)
    print(f"Collision Rate: {metrics['collision']/num_episodes*100:.2f}%", flush=True)
    print(f"Timeout Rate: {metrics['timeout']/num_episodes*100:.2f}%", flush=True)
    print(f"Mean Episode Length: {np.mean(metrics['lengths']):.2f} ± {np.std(metrics['lengths']):.2f}", flush=True)
    print(f"Mean Return: {np.mean(metrics['returns']):.2f} ± {np.std(metrics['returns']):.2f}", flush=True)
    print(f"Mean Path Length: {np.mean(metrics['path_lengths']):.2f} ± {np.std(metrics['path_lengths']):.2f}", flush=True)
    print(f"Results saved to: {eval_csv_path}", flush=True)
    print("="*60, flush=True)

def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def main(args):
    rospy.init_node('ppo_stage_1')
    
    # Setup vision mode based on method_name
    use_vision = 'vision' in args.method_name.lower()
    vision_dim = 64  # Default vision feature dimension
    
    if use_vision:
        print(f"[main] Vision mode enabled for method: {args.method_name}", flush=True)
    
    env = Env(is_training, use_vision=use_vision, vision_dim=vision_dim)
    
    # Calculate actual state dimension
    actual_state_dim = state_dim + (vision_dim if use_vision else 0)
    print(f'State Dimensions: {actual_state_dim} (base={state_dim}, vision={vision_dim if use_vision else 0})', flush=True)
    # agent = DDPG(env, state_dim, action_dim)


    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')
    """
        The main function to run.

        Parameters:
            args - the arguments parsed from command line

        Return:
            None
    """
    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
        'timesteps_per_batch': args.timesteps_per_episode,
        'max_timesteps_per_episode': args.timesteps_per_episode,
        'gamma': 0.99,
        'n_updates_per_iteration': 50,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10,
        'exp_id': "v02_simple_env_60_reward_proportion",
        'method_name': args.method_name,
        'state_dim': actual_state_dim,
        'output_dir': args.output_dir
    }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym and have both continuous
    # observation and action spaces.
    # Train or test, depending on the mode specified
    if args.eval:
        # Evaluation mode
        evaluate(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, 
                 critic_model=args.critic_model, num_episodes=args.eval_episodes)
    elif args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model,
              critic_model=args.critic_model, max_timesteps=args.max_timesteps)
        ### env.logger_global
    else:
        test(env=env, actor_model=args.actor_model)


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    main(args)
