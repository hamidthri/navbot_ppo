#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import print_function
import gym
import sys
import torch
import json
import subprocess
import platform

from arguments import get_args
from ppo import PPO
from net_actor import NetActor, ViTFiLMTokenLearnerActor, RecurrentViTFiLMTokenLearnerActor
from net_critic import NetCritic, ViTFiLMTokenLearnerCritic, RecurrentViTFiLMTokenLearnerCritic
from eval_policy import eval_policy
import numpy as np

import os, glob


state_dim = 16
action_dim = 2
action_linear_max = 0.25  # m/s
action_angular_max = 1  # rad/s
is_training = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_sampler_mode(args):
    """
    Visualize sampler by cycling through samples in Gazebo.
    Keeps Gazebo running, does not start training.
    """
    from rect_region_sampler import visualize_samples_in_gazebo
    
    n_samples = args.visualize_sampler
    delay_sec = args.viz_delay_sec
    
    print(f"\n{'='*70}")
    print(f"VISUALIZE SAMPLER MODE")
    print(f"{'='*70}")
    print(f"Sampler mode: {args.sampler_mode}")
    print(f"Samples: {n_samples}")
    print(f"Delay: {delay_sec}s")
    print(f"{'='*70}\n")
    
    if args.sampler_mode != 'rect_regions':
        print(f"ERROR: --visualize_sampler only works with --sampler_mode rect_regions")
        print(f"Current sampler_mode: {args.sampler_mode}")
        sys.exit(1)
    
    visualize_samples_in_gazebo(n_samples=n_samples, delay_sec=delay_sec)


def get_architecture_classes(architecture_name):
    """
    Get actor and critic classes based on architecture name.
    
    Args:
        architecture_name: One of ['default', 'vit_film_tokenlearner', 'recurrent_vit_film_tokenlearner']
    
    Returns:
        (actor_class, critic_class)
    """
    if architecture_name == 'vit_film_tokenlearner':
        return ViTFiLMTokenLearnerActor, ViTFiLMTokenLearnerCritic
    elif architecture_name == 'recurrent_vit_film_tokenlearner':
        return RecurrentViTFiLMTokenLearnerActor, RecurrentViTFiLMTokenLearnerCritic
    elif architecture_name == 'default':
        return NetActor, NetCritic
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

def save_run_metadata(run_dir, args, hyperparameters):
    """Save comprehensive run metadata for benchmarking and reproducibility"""
    metadata = {}
    
    # Run identification
    metadata['run_name'] = args.run_name if args.run_name else args.method_name
    metadata['timestamp'] = hyperparameters.get('method_name', 'unknown').split('_')[0] if '_' in hyperparameters.get('method_name', '') else 'unknown'
    
    # Git information
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        git_dirty = subprocess.call(['git', 'diff-index', '--quiet', 'HEAD'], stderr=subprocess.DEVNULL) != 0
        metadata['git_commit'] = git_commit
        metadata['git_dirty'] = git_dirty
    except:
        metadata['git_commit'] = 'unknown'
        metadata['git_dirty'] = 'unknown'
    
    # Environment information
    metadata['hostname'] = platform.node()
    metadata['os'] = f"{platform.system()} {platform.release()}"
    metadata['python_version'] = platform.python_version()
    
    # PyTorch information
    metadata['torch_version'] = torch.__version__
    metadata['torch_cuda_version'] = torch.version.cuda if torch.cuda.is_available() else 'N/A'
    metadata['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        metadata['gpu_name'] = torch.cuda.get_device_name(0)
        metadata['gpu_capability'] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
        metadata['cuda_arch_list'] = torch.cuda.get_arch_list()
    
    # Training configuration
    metadata['vision_backbone'] = args.vision_backbone
    metadata['vision_proj_dim'] = args.vision_proj_dim
    metadata['use_map_sampler'] = args.use_map_sampler
    metadata['distance_uniform'] = args.distance_uniform
    metadata['reward_type'] = args.reward_type
    metadata['timesteps_per_episode'] = args.timesteps_per_episode
    metadata['max_timesteps'] = args.max_timesteps
    metadata['steps_per_iteration'] = args.steps_per_iteration
    metadata['save_every_iterations'] = args.save_every_iterations
    
    # Paths
    metadata['run_dir'] = run_dir
    metadata['checkpoint_dir'] = os.path.join(run_dir, 'checkpoints')
    metadata['log_dir'] = os.path.join(run_dir, 'logs')
    metadata['tensorboard_dir'] = os.path.join(run_dir, 'tb')
    
    # Save as JSON
    meta_path = os.path.join(run_dir, 'run_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[Metadata] Saved run metadata: {meta_path}", flush=True)
    print(f"[Metadata] Git commit: {metadata['git_commit'][:8]}... (dirty: {metadata['git_dirty']})", flush=True)
    print(f"[Metadata] GPU: {metadata.get('gpu_name', 'N/A')} (capability: {metadata.get('gpu_capability', 'N/A')})", flush=True)
    return meta_path



def train(env, hyperparameters, actor_model, critic_model, max_timesteps=5000, args=None):
    """
        Trains the model.

        Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training
            max_timesteps - maximum timesteps to train for
            args - command line arguments for metadata saving

        Return:
            None
    """
    print(f"Start Training ... ", flush=True)
    
    # Get actual state_dim from environment (may include vision features)
    actual_state_dim = hyperparameters.pop('state_dim', state_dim)
    
    # Get architecture classes based on args
    architecture_name = args.architecture if args is not None else 'default'
    actor_class, critic_class = get_architecture_classes(architecture_name)
    
    # Pass architecture-specific hyperparameters if needed
    arch_hyperparams = {}
    if architecture_name in ['vit_film_tokenlearner', 'recurrent_vit_film_tokenlearner']:
        arch_hyperparams['num_learned_tokens'] = args.num_learned_tokens if args else 8
        arch_hyperparams['vision_emb_dim'] = args.vision_emb_dim if args else 128
        if architecture_name == 'recurrent_vit_film_tokenlearner':
            arch_hyperparams['gru_hidden_dim'] = args.gru_hidden_dim if args else 128
    
    print(f"[Architecture] Using: {architecture_name}", flush=True)
    print(f"[Architecture] Actor: {actor_class.__name__}, Critic: {critic_class.__name__}", flush=True)

    # Create a model for PPO.
    agent = PPO(policy_class=actor_class, value_func=critic_class, env=env, state_dim=actual_state_dim, action_dim=action_dim,
                **hyperparameters, **arch_hyperparams)
    
    # Save run metadata for benchmarking
    if args is not None:
        save_run_metadata(agent.method_run_dir, args, hyperparameters)
    
    past_action = np.array([0., 0.])

    # Tries to load in an existing actor/critic model to continue training on
    checkpoint_pattern = os.path.join(agent.checkpoint_dir, 'critic_iter*_step*.pth')
    critic_paths = sorted(glob.glob(checkpoint_pattern))
    actor_pattern = os.path.join(agent.checkpoint_dir, 'actor_iter*_step*.pth')
    actor_paths = sorted(glob.glob(actor_pattern))
    
    # Check if resume is requested
    resume = hyperparameters.get('resume', False)
    
    # Only load if resume=True and state_dim matches
    load_checkpoint = False
    if resume and critic_paths != [] and actor_paths != []:
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
                            print(f"[Resume Mode] Checkpoint state_dim matches ({checkpoint_input_dim}), will load", flush=True)
                        else:
                            print(f"[Resume Mode] Checkpoint state_dim mismatch: checkpoint={checkpoint_input_dim}, current={actual_state_dim}", flush=True)
                            print(f"Training from scratch with new state_dim", flush=True)
                        break
        except Exception as e:
            print(f"Could not check checkpoint dimensions: {e}", flush=True)
    elif not resume and (critic_paths != [] or actor_paths != []):
        print(f"[Fresh Training] Existing checkpoints found but --resume not specified. Training from scratch.", flush=True)
    
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

def evaluate(env, hyperparameters, actor_model, critic_model, num_episodes, stochastic=False):
    """
        Evaluation mode: Load checkpoint and run N episodes to compute metrics.
        
        Args:
            stochastic: If True, sample actions from policy distribution (training-like).
                       If False, use mean action (deterministic eval).
    """
    import csv
    import time as time_module
    from torch.distributions import MultivariateNormal
    
    eval_mode = "STOCHASTIC" if stochastic else "DETERMINISTIC"
    print(f"\nEvaluation: Running {num_episodes} episodes in {eval_mode} mode", flush=True)
    
    method_name = hyperparameters.get('method_name', 'baseline')
    exp_id = hyperparameters['exp_id']
    actual_state_dim = hyperparameters.pop('state_dim', state_dim)
    
    # Get output directory from args
    import arguments
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs')
    output_dir = hyperparameters.get('output_dir', default_output)
    
    # Load model
    if actor_model == '':
        # Try to find latest checkpoint in new structure (support both old and new naming)
        ckpt_dir = os.path.join(output_dir, method_name, 'checkpoints')
        actor_path = sorted(glob.glob(os.path.join(ckpt_dir, f'actor_iter*_step*.pth')))
        if not actor_path:
            # Fallback to old naming
            actor_path = sorted(glob.glob(os.path.join(ckpt_dir, f'actor_step*.pth')))
        if not actor_path:
            print("No checkpoint found for evaluation. Exiting.", flush=True)
            sys.exit(0)
        actor_model = actor_path[-1]
    
    print("\n" + "="*80, flush=True)
    print("CHECKPOINT LOADING", flush=True)
    print("="*80, flush=True)
    print(f"Checkpoint path: {actor_model}", flush=True)
    print(f"Checkpoint exists: {os.path.exists(actor_model)}", flush=True)
    print(f"Checkpoint size: {os.path.getsize(actor_model) / (1024*1024):.2f} MB", flush=True)
    print("="*80 + "\n", flush=True)
    
    # Extract vision and architecture parameters from hyperparameters
    vision_backbone = hyperparameters.get('vision_backbone', None)
    vision_proj_dim = hyperparameters.get('vision_proj_dim', 64)
    architecture_name = hyperparameters.get('architecture', 'default')
    
    # Build actor with same architecture as training
    arch_hyperparams = {}
    if vision_backbone:
        print(f"[Eval] Vision enabled: backbone={vision_backbone}, proj_dim={vision_proj_dim}, arch={architecture_name}", flush=True)
        arch_hyperparams['use_vision'] = True
        
        # Get vision feature dim based on backbone
        from vision_backbones import get_backbone_feat_dim
        vision_feat_dim = get_backbone_feat_dim(vision_backbone)
        arch_hyperparams['vision_feat_dim'] = vision_feat_dim
        arch_hyperparams['vision_proj_dim'] = vision_proj_dim
        
        if architecture_name == 'vit_film_tokenlearner' or architecture_name == 'recurrent_vit_film_tokenlearner':
            arch_hyperparams['num_learned_tokens'] = hyperparameters.get('num_learned_tokens', 8)
            arch_hyperparams['vision_emb_dim'] = hyperparameters.get('vision_emb_dim', 128)
            if architecture_name == 'recurrent_vit_film_tokenlearner':
                arch_hyperparams['gru_hidden_dim'] = hyperparameters.get('gru_hidden_dim', 128)
    
    policy = NetActor(actual_state_dim, action_dim, **arch_hyperparams).to(device)
    policy.load_state_dict(torch.load(actor_model))
    policy.eval()
    
    # Prepare CSV in new structure
    log_dir = os.path.join(output_dir, method_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    eval_csv_path = os.path.join(log_dir, f'{method_name}_eval_episodes.csv')
    
    with open(eval_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'success', 'collision', 'timeout', 'length', 'return', 'path_length', 'time'])
    
    # Run episodes
    metrics = {'success': 0, 'collision': 0, 'timeout': 0, 'lengths': [], 'returns': [], 'path_lengths': [], 'times': []}
    
    # Track action statistics for diagnosis
    all_actions_linear = []
    all_actions_angular = []
    all_cmd_vel_linear = []
    all_cmd_vel_angular = []
    
    # Stochastic sampling setup (similar to PPO training)
    if stochastic:
        # Use same covariance as training for consistency
        cov_var = torch.full(size=(action_dim,), fill_value=0.5)
        cov_mat = torch.diag(cov_var).to(device)
    
    print("\n" + "="*80, flush=True)
    print(f"STARTING EVALUATION: {num_episodes} EPISODES ({eval_mode} MODE)", flush=True)
    print(f"Action bounds: linear [0, 1] → cmd_vel [0, {action_linear_max}] m/s", flush=True)
    print(f"               angular [-1, 1] → cmd_vel [-{action_angular_max}, {action_angular_max}] rad/s", flush=True)
    if stochastic:
        print(f"Stochastic mode: sampling from policy distribution with cov_var={cov_var[0]:.3f}", flush=True)
    print("="*80 + "\n", flush=True)
    
    for ep in range(num_episodes):
        ep_start_time = time_module.time()
        obs = env.reset()
        done = False
        arrive = False
        ep_return = 0
        ep_length = 0
        path_length = 0.0
        past_action = np.array([0., 0.])
        prev_pos = None
        
        # Print first few actions for first episode
        print_actions = (ep == 0)
        
        while ep_length < hyperparameters['max_timesteps_per_episode']:
            # Action selection: deterministic (mean) or stochastic (sampled)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # Add batch dimension
                mean_action = policy(obs_tensor).squeeze(0)  # Keep on device for distribution
                
                if stochastic:
                    # Sample from distribution (training-like)
                    dist = MultivariateNormal(mean_action, cov_mat)
                    action_tensor = dist.sample()
                    
                    # Clamp to valid ranges
                    action_clamped = torch.stack([
                        torch.clamp(action_tensor[0], 0, 1),  # linear
                        torch.clamp(action_tensor[1], -1, 1)  # angular
                    ])
                    action_raw = action_clamped.cpu().numpy()
                    
                    # Track distribution params for diagnosis
                    if print_actions and ep_length < 50:
                        mean_np = mean_action.cpu().numpy()
                        sampled_np = action_tensor.cpu().numpy()
                else:
                    # Deterministic: use mean
                    action_raw = mean_action.cpu().numpy()
            
            # The actor outputs: linear in [0,1], angular in [-1,1]
            # These are already the correct ranges, no further scaling needed before env.step
            action = action_raw.copy()
            
            # Compute cmd_vel (env divides linear by 4, which equals *0.25)
            cmd_vel_linear = action[0] / 4  # This is what env.step does
            cmd_vel_angular = action[1]
            
            # Track actions for statistics
            all_actions_linear.append(action[0])
            all_actions_angular.append(action[1])
            all_cmd_vel_linear.append(cmd_vel_linear)
            all_cmd_vel_angular.append(cmd_vel_angular)
            
            # Print diagnostic info for first 50 steps of first episode
            if print_actions and ep_length < 50:
                if ep_length < 5 or ep_length % 10 == 0:
                    if stochastic:
                        print(f"Step {ep_length:3d}: mean=[{mean_np[0]:.6f}, {mean_np[1]:.6f}] "
                              f"sampled=[{sampled_np[0]:.6f}, {sampled_np[1]:.6f}] "
                              f"clamped=[{action[0]:.6f}, {action[1]:.6f}] → "
                              f"cmd_vel=[{cmd_vel_linear:.6f} m/s, {cmd_vel_angular:.6f} rad/s]", flush=True)
                    else:
                        print(f"Step {ep_length:3d}: action=[{action[0]:.6f}, {action[1]:.6f}] → "
                              f"cmd_vel=[{cmd_vel_linear:.6f} m/s, {cmd_vel_angular:.6f} rad/s]", flush=True)
            
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
        
        ep_time = time_module.time() - ep_start_time
        
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
        metrics['times'].append(ep_time)
        
        # Write to CSV
        with open(eval_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, success, collision, timeout, ep_length, ep_return, path_length, ep_time])
        
        if (ep + 1) % 10 == 0:
            print(f"Evaluated {ep+1}/{num_episodes} episodes", flush=True)
    
    # Print summary
    print("\n" + "="*80, flush=True)
    print("EVALUATION SUMMARY", flush=True)
    print("="*80, flush=True)
    print(f"Checkpoint: {os.path.basename(actor_model)}", flush=True)
    print(f"Method: {method_name}", flush=True)
    print(f"Episodes: {num_episodes}", flush=True)
    print(f"\nOutcomes:", flush=True)
    print(f"  Success Rate:   {metrics['success']/num_episodes*100:.2f}% ({metrics['success']}/{num_episodes})", flush=True)
    print(f"  Collision Rate: {metrics['collision']/num_episodes*100:.2f}% ({metrics['collision']}/{num_episodes})", flush=True)
    print(f"  Timeout Rate:   {metrics['timeout']/num_episodes*100:.2f}% ({metrics['timeout']}/{num_episodes})", flush=True)
    print(f"\nEpisode Metrics:", flush=True)
    print(f"  Mean Episode Length: {np.mean(metrics['lengths']):.2f} ± {np.std(metrics['lengths']):.2f}", flush=True)
    print(f"  Mean Return: {np.mean(metrics['returns']):.2f} ± {np.std(metrics['returns']):.2f}", flush=True)
    print(f"  Mean Path Length: {np.mean(metrics['path_lengths']):.2f} ± {np.std(metrics['path_lengths']):.2f} m", flush=True)
    print(f"  Mean Episode Time: {np.mean(metrics['times']):.2f}s ± {np.std(metrics['times']):.2f}s", flush=True)
    
    # Action statistics for diagnosis
    print(f"\nAction Statistics (all {len(all_actions_linear)} steps):", flush=True)
    print(f"  Linear velocity (network output [0,1]):", flush=True)
    print(f"    Mean: {np.mean(all_actions_linear):.6f}", flush=True)
    print(f"    Std:  {np.std(all_actions_linear):.6f}", flush=True)
    print(f"    Min:  {np.min(all_actions_linear):.6f}", flush=True)
    print(f"    Max:  {np.max(all_actions_linear):.6f}", flush=True)
    print(f"    Fraction < 0.01: {np.mean(np.array(all_actions_linear) < 0.01)*100:.2f}%", flush=True)
    
    print(f"\n  CMD_VEL Linear (m/s, after /4 scaling):", flush=True)
    print(f"    Mean: {np.mean(all_cmd_vel_linear):.6f} m/s", flush=True)
    print(f"    Std:  {np.std(all_cmd_vel_linear):.6f} m/s", flush=True)
    print(f"    Min:  {np.min(all_cmd_vel_linear):.6f} m/s", flush=True)
    print(f"    Max:  {np.max(all_cmd_vel_linear):.6f} m/s", flush=True)
    print(f"    Fraction > 0.02: {np.mean(np.array(all_cmd_vel_linear) > 0.02)*100:.2f}% (meaningful motion)", flush=True)
    print(f"    Fraction > 0.10: {np.mean(np.array(all_cmd_vel_linear) > 0.10)*100:.2f}% (strong motion)", flush=True)
    
    print(f"\n  Angular velocity (network output [-1,1]):", flush=True)
    print(f"    Mean: {np.mean(all_actions_angular):.6f}", flush=True)
    print(f"    Std:  {np.std(all_actions_angular):.6f}", flush=True)
    print(f"    Min:  {np.min(all_actions_angular):.6f}", flush=True)
    print(f"    Max:  {np.max(all_actions_angular):.6f}", flush=True)
    
    print(f"\n  CMD_VEL Angular (rad/s, no scaling):", flush=True)
    print(f"    Mean(abs): {np.mean(np.abs(all_cmd_vel_angular)):.6f} rad/s", flush=True)
    print(f"    Std:  {np.std(all_cmd_vel_angular):.6f} rad/s", flush=True)
    print(f"    Fraction |w| > 0.2: {np.mean(np.abs(all_cmd_vel_angular) > 0.2)*100:.2f}% (turning)", flush=True)
    print(f"    Fraction |w| > 0.5: {np.mean(np.abs(all_cmd_vel_angular) > 0.5)*100:.2f}% (strong turning)", flush=True)
    
    print(f"\nResults saved to: {eval_csv_path}", flush=True)
    print("="*80 + "\n", flush=True)

def makepath(desired_path, isfile=False):
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

def dry_run_vision_pipeline(args):
    """
    Dry-run mode to test vision pipeline without Gazebo.
    Tests: image loading -> feature extraction -> state concat -> network forward -> checkpoint save
    """
    print("\n" + "="*60)
    print("[DRY_RUN] Vision Pipeline Test (No Gazebo)")
    print("="*60 + "\n")
    
    use_vision = True
    vision_dim = 64
    actual_state_dim = state_dim + vision_dim
    
    # Setup output directories
    from ppo import makepath as ppo_makepath
    method_run_dir = os.path.join(args.output_dir, args.method_name)
    checkpoint_dir = os.path.join(method_run_dir, 'checkpoints')
    log_dir = os.path.join(method_run_dir, 'logs')
    frame_dir = os.path.join(log_dir, 'dryrun_frames')
    
    ppo_makepath(checkpoint_dir)
    ppo_makepath(log_dir)
    ppo_makepath(frame_dir)
    
    print(f"[DRY_RUN] Output directory: {method_run_dir}")
    print(f"[DRY_RUN] Checkpoints: {checkpoint_dir}")
    print(f"[DRY_RUN] Logs: {log_dir}")
    print(f"[DRY_RUN] Frames: {frame_dir}\n")
    
    # Initialize vision encoder
    print("[DRY_RUN] Loading vision encoder...")
    from vision_encoder import VisionEncoder
    vision_encoder = VisionEncoder(output_dim=vision_dim, use_pretrained=True)
    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()
    print(f"[DRY_RUN] Vision encoder loaded on {device}\n")
    
    # Generate or load test images
    print("[DRY_RUN] Generating test images...")
    import PIL.Image
    test_images = []
    for i in range(5):
        # Create a simple test pattern (gradient)
        img_arr = np.zeros((240, 320, 3), dtype=np.uint8)
        # Create a gradient pattern that varies per frame
        for y in range(240):
            for x in range(320):
                img_arr[y, x, 0] = (x + i * 50) % 256  # Red channel
                img_arr[y, x, 1] = (y + i * 30) % 256  # Green channel
                img_arr[y, x, 2] = ((x + y) + i * 20) % 256  # Blue channel
        
        img = PIL.Image.fromarray(img_arr, 'RGB')
        frame_path = os.path.join(frame_dir, f'test_frame_{i:03d}.png')
        img.save(frame_path)
        test_images.append(img_arr)
        print(f"[DRY_RUN] Saved frame {i}: {frame_path}")
    
    print()
    
    # Extract vision features from test images
    print("[DRY_RUN] Extracting vision features...")
    import torchvision.transforms as transforms
    vision_features_list = []
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for i, img_arr in enumerate(test_images):
        img_tensor = transform(img_arr).unsqueeze(0).to(device)
        with torch.no_grad():
            features = vision_encoder(img_tensor)
        vision_features_list.append(features.cpu().numpy()[0])
        print(f"[DRY_RUN] Frame {i}: features shape={features.shape}, "
              f"range=[{features.min():.3f}, {features.max():.3f}], "
              f"mean={features.mean():.3f}, norm={torch.norm(features).item():.3f}")
    
    print()
    
    # Create dummy state vectors
    print("[DRY_RUN] Creating dummy state vectors...")
    dummy_states = []
    for i, vision_feats in enumerate(vision_features_list):
        # LiDAR part: 10 normalized scan values
        lidar = np.random.uniform(0.1, 1.0, size=10)
        # Goal info: past_action (2) + rel_dis, yaw, rel_theta, diff_angle (4)
        past_action = np.array([0.1, 0.05])
        goal_info = np.array([0.5, 0.3, 0.2, 0.1])  # normalized values
        
        # Concatenate: lidar + past_action + goal_info + vision
        state = np.concatenate([lidar, past_action, goal_info, vision_feats])
        dummy_states.append(state)
        print(f"[DRY_RUN] State {i}: shape={state.shape}, "
              f"lidar={len(lidar)}, past_action={len(past_action)}, "
              f"goal={len(goal_info)}, vision={len(vision_feats)}, total={len(state)}")
    
    print(f"\n[DRY_RUN] Expected state_dim: {actual_state_dim}, Actual: {len(dummy_states[0])}")
    assert len(dummy_states[0]) == actual_state_dim, f"State dimension mismatch!"
    print("[DRY_RUN] ✓ State dimensions correct\n")
    
    # Initialize networks
    print("[DRY_RUN] Initializing actor and critic networks...")
    actor = NetActor(actual_state_dim, action_dim).to(device)
    critic = NetCritic(actual_state_dim, 1).to(device)
    print(f"[DRY_RUN] Actor parameters: {sum(p.numel() for p in actor.parameters())}")
    print(f"[DRY_RUN] Critic parameters: {sum(p.numel() for p in critic.parameters())}\n")
    
    # Test forward passes
    print("[DRY_RUN] Testing forward passes...")
    for i, state in enumerate(dummy_states):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_mean = actor(state_tensor)
            value = critic(state_tensor)
        
        print(f"[DRY_RUN] Step {i}: action_mean={action_mean[0].cpu().numpy()}, "
              f"value={value.item():.3f}")
    
    print("[DRY_RUN] ✓ All forward passes successful\n")
    
    # Save checkpoints
    print("[DRY_RUN] Saving checkpoints...")
    actor_path = os.path.join(checkpoint_dir, f'actor_dryrun.pth')
    critic_path = os.path.join(checkpoint_dir, f'critic_dryrun.pth')
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    print(f"[DRY_RUN] Saved actor: {actor_path}")
    print(f"[DRY_RUN] Saved critic: {critic_path}\n")
    
    # Save CSV log
    print("[DRY_RUN] Saving dry-run log...")
    import csv
    csv_path = os.path.join(log_dir, f'{args.method_name}_dryrun.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['status', 'frames_saved', 'state_dim', 'actor_params', 'critic_params'])
        writer.writerow(['success', len(test_images), actual_state_dim, 
                        sum(p.numel() for p in actor.parameters()),
                        sum(p.numel() for p in critic.parameters())])
    print(f"[DRY_RUN] Saved log: {csv_path}\n")
    
    print("="*60)
    print("[DRY_RUN] ✓ Vision pipeline test completed successfully!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Frames saved: {len(test_images)} in {frame_dir}")
    print(f"  - State dim: {actual_state_dim} (16 base + 64 vision)")
    print(f"  - Forward passes: {len(dummy_states)} successful")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Log CSV: {csv_path}")
    print()

def main(args):
    import rospy
    rospy.init_node('ppo_stage_1')
    
    # Import Env after rospy is initialized
    from environment_new import Env
    
    # Setup vision mode based on vision_backbone argument
    use_vision = args.vision_backbone is not None and args.vision_backbone != 'none'
    vision_dim = args.vision_proj_dim if hasattr(args, 'vision_proj_dim') else 64
    
    if use_vision:
        print(f"[main] Vision mode enabled: backbone={args.vision_backbone}, proj_dim={vision_dim}", flush=True)
    else:
        print(f"[main] Vision mode DISABLED (no backbone specified)", flush=True)
    
    # Determine sampler mode (support both new and legacy flags)
    sampler_mode = args.sampler_mode
    if args.use_map_sampler and sampler_mode == 'map':
        # Already correct
        pass
    elif args.use_map_sampler and sampler_mode != 'map':
        # Legacy flag override
        sampler_mode = 'map'
    elif args.use_external_sampler and sampler_mode != 'external':
        # Legacy flag override
        sampler_mode = 'external'
    
    # Compute method_run_dir before creating env (needed for fixed case capture)
    method_run_dir = os.path.join(args.output_dir, args.method_name)
    
    env = Env(is_training, use_vision=use_vision, vision_dim=vision_dim, 
              sampler_mode=sampler_mode, debug_sampler=args.debug_sampler,
              distance_uniform=args.distance_uniform, reward_type=args.reward_type,
              fixed_case_path=args.fixed_case_path, method_run_dir=method_run_dir,
              curriculum_min_dist=args.curriculum_min_dist,
              curriculum_max_dist=args.curriculum_max_dist,
              curriculum_steps=args.curriculum_steps)
    
    # State dimension is ONLY base state (LiDAR + pose + past actions)
    # Vision features are handled separately in policy network
    actual_state_dim = state_dim  # Always 16 for base state
    print(f'[main] Base State Dimensions: {actual_state_dim}', flush=True)
    if use_vision:
        print(f'[main] Vision mode enabled: features extracted in policy network (not concatenated to state)', flush=True)
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
    
    # Use run_name if provided, otherwise fall back to method_name
    effective_method_name = args.run_name if args.run_name else args.method_name
    
    hyperparameters = {
        'timesteps_per_batch': args.steps_per_iteration,
        'max_timesteps_per_episode': args.timesteps_per_episode,
        'gamma': 0.99,
        'n_updates_per_iteration': 50,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10,
        'save_freq': args.save_every_iterations,
        'exp_id': "v02_simple_env_60_reward_proportion",
        'method_name': effective_method_name,
        'state_dim': actual_state_dim,
        'output_dir': args.output_dir,
        'resume': args.resume,
        'vision_backbone': args.vision_backbone,
        'vision_proj_dim': args.vision_proj_dim,
    }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym and have both continuous
    # observation and action spaces.
    # Train or test, depending on the mode specified
    if args.eval:
        # Evaluation mode
        evaluate(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, 
                 critic_model=args.critic_model, num_episodes=args.eval_episodes, 
                 stochastic=args.eval_stochastic)
    elif args.visualize_sampler > 0:
        # Sampler visualization mode (no training)
        visualize_sampler_mode(args)
    elif args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model,
              critic_model=args.critic_model, max_timesteps=args.max_timesteps, args=args)
        ### env.logger_global
    else:
        test(env=env, actor_model=args.actor_model)


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    main(args)
