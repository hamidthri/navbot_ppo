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
from net_actor import NetActor
from net_critic import NetCritic
from eval_policy import eval_policy
import numpy as np

import os, glob


state_dim = 16
action_dim = 2
action_linear_max = 0.25  # m/s
action_angular_max = 1  # rad/s
is_training = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Create a model for PPO.
    agent = PPO(policy_class=NetActor, value_func=NetCritic, env=env, state_dim=actual_state_dim, action_dim=action_dim,
                **hyperparameters)
    
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
    
    print(f"\nEvaluation: Running {num_episodes} episodes", flush=True)
    
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
    
    print(f"Loading actor: {actor_model}", flush=True)
    
    policy = NetActor(actual_state_dim, action_dim).to(device)
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
    print(f"Mean Episode Time: {np.mean(metrics['times']):.2f}s ± {np.std(metrics['times']):.2f}s", flush=True)
    print(f"Results saved to: {eval_csv_path}", flush=True)
    print("="*60, flush=True)

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
    # Check for dry-run mode BEFORE initializing ROS
    if args.dry_run_vision:
        dry_run_vision_pipeline(args)
        return
    
    import rospy
    rospy.init_node('ppo_stage_1')
    
    # Import Env after rospy is initialized
    from environment_new import Env
    
    # Setup vision mode based on method_name
    use_vision = 'vision' in args.method_name.lower()
    vision_dim = 64  # Default vision feature dimension
    
    if use_vision:
        print(f"[main] Vision mode enabled for method: {args.method_name}", flush=True)
    
    env = Env(is_training, use_vision=use_vision, vision_dim=vision_dim, 
              use_map_sampler=args.use_map_sampler, debug_sampler=args.debug_sampler,
              distance_uniform=args.distance_uniform, reward_type=args.reward_type)
    
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
                 critic_model=args.critic_model, num_episodes=args.eval_episodes)
    elif args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model,
              critic_model=args.critic_model, max_timesteps=args.max_timesteps, args=args)
        ### env.logger_global
    else:
        test(env=env, actor_model=args.actor_model)


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    main(args)
