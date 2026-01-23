"""
        This file contains the arguments to parse at command line.
        File main.py will call get_args, which then the arguments
        will be returned.
"""
import argparse
import os

def get_args():
        """
                Description:
                Parses arguments at command line.

                Parameters:
                        None

                Return:
                        args - the arguments parsed
        """
        parser = argparse.ArgumentParser()

        parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
        parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
        parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename
        parser.add_argument('--method_name', dest='method_name', type=str, default='baseline')  # method identifier for checkpoints/logs
        parser.add_argument('--eval', dest='eval', action='store_true', default=False)     # evaluation mode flag
        parser.add_argument('--eval_episodes', dest='eval_episodes', type=int, default=100)  # number of episodes for evaluation
        parser.add_argument('--output_dir', dest='output_dir', type=str, default=None)     # base output directory for runs
        parser.add_argument('--timesteps_per_episode', dest='timesteps_per_episode', type=int, default=500)  # max timesteps per episode
        parser.add_argument('--max_timesteps', dest='max_timesteps', type=int, default=5000)  # total training timesteps
        parser.add_argument('--steps_per_iteration', dest='steps_per_iteration', type=int, default=5000)  # timesteps per PPO iteration (batch)
        parser.add_argument('--save_every_iterations', dest='save_every_iterations', type=int, default=2)  # save checkpoint every N iterations
        parser.add_argument('--resume', dest='resume', action='store_true', default=False)  # resume from existing checkpoint
        parser.add_argument('--dry_run_vision', dest='dry_run_vision', action='store_true', default=False)  # dry-run mode without Gazebo
        parser.add_argument('--vision_backbone', dest='vision_backbone', type=str, default='mobilenet_v2',
                            choices=['mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'clip_vit_b32'],
                            help='Vision backbone architecture')
        parser.add_argument('--vision_proj_dim', dest='vision_proj_dim', type=int, default=64, help='Vision projection dimension')
        
        # NEW: External sampler flag
        parser.add_argument('--use_external_sampler', dest='use_external_sampler', action='store_true', default=False,
                            help='Use external spawn_goal_sampler for open-space start/goal (default: OFF)')
        
        # NEW: Map-based sampler flag (with distance transform)
        parser.add_argument('--use_map_sampler', dest='use_map_sampler', action='store_true', default=False,
                            help='Use map-based sampler with distance transform for collision-free start/goal (default: OFF)')
        
        # NEW: Debug sampler flag (for noisy reset logs)
        parser.add_argument('--debug_sampler', dest='debug_sampler', action='store_true', default=False,
                            help='Print detailed reset logs to console (default: OFF)')
        
        # NEW: Distance-uniform sampling flag
        parser.add_argument('--distance_uniform', dest='distance_uniform', action='store_true', default=False,
                            help='Use distance-uniform goal sampling (sample distance bin first, then goal from bin) instead of spatial-uniform (default: OFF)')
        
        # NEW: Tiny debug run flag
        parser.add_argument('--tiny_debug_run', dest='tiny_debug_run', action='store_true', default=False,
                            help='Fast debug mode: episode_max_steps=20, steps_per_iter=40, max_timesteps=400')
        
        args = parser.parse_args()

        # Set default output_dir to repo_root/runs if not specified
        if args.output_dir is None:
                # Get repo root (3 levels up from src/ directory)
                src_dir = os.path.dirname(os.path.abspath(__file__))
                repo_root = os.path.abspath(os.path.join(src_dir, '...', '...', '...'))
                args.output_dir = os.path.join(repo_root, 'runs')
        
        # Apply tiny_debug_run overrides
        if args.tiny_debug_run:
                print("[arguments] TINY_DEBUG_RUN mode activated:", flush=True)
                print("  - timesteps_per_episode: 500 -> 20", flush=True)
                print("  - steps_per_iteration: 5000 -> 40", flush=True)
                print("  - max_timesteps: 5000 -> 400", flush=True)
                args.timesteps_per_episode = 20
                args.steps_per_iteration = 40
                args.max_timesteps = 400

        return args
