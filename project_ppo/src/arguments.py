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
        parser.add_argument('--run_name', dest='run_name', type=str, default=None)  # standardized run name (overrides method_name)
        parser.add_argument('--eval', dest='eval', action='store_true', default=False)     # evaluation mode flag
        parser.add_argument('--eval_episodes', dest='eval_episodes', type=int, default=100)  # number of episodes for evaluation
        parser.add_argument('--eval_stochastic', dest='eval_stochastic', action='store_true', default=False,
                            help='Use stochastic action sampling in eval (training-like) instead of deterministic mean')
        parser.add_argument('--output_dir', dest='output_dir', type=str, default=None)     # base output directory for runs
        parser.add_argument('--load_config', dest='load_config', type=str, default=None)   # load config from baseline run (path to config.yml)
        parser.add_argument('--runs_root', dest='runs_root', type=str, default='runs')     # runs directory name (default: runs, test: runs_test)
        parser.add_argument('--timesteps_per_episode', dest='timesteps_per_episode', type=int, default=500)  # max timesteps per episode
        parser.add_argument('--max_timesteps', dest='max_timesteps', type=int, default=5000)  # total training timesteps
        parser.add_argument('--steps_per_iteration', dest='steps_per_iteration', type=int, default=5000)  # timesteps per PPO iteration (batch)
        parser.add_argument('--save_every_iterations', dest='save_every_iterations', type=int, default=2)  # save checkpoint every N iterations
        parser.add_argument('--resume', dest='resume', action='store_true', default=False)  # resume from existing checkpoint
        parser.add_argument('--vision_backbone', dest='vision_backbone', type=str, default='resnet18',
                            choices=['mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'clip_vit_b32', 
                                     'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                            help='Vision backbone architecture')
        parser.add_argument('--vision_proj_dim', dest='vision_proj_dim', type=int, default=64, help='Vision projection dimension')
        
        # NEW: Architecture selection
        parser.add_argument('--architecture', dest='architecture', type=str, default='default',
                            choices=['default', 'vit_film_tokenlearner', 'recurrent_vit_film_tokenlearner'],
                            help='Policy architecture: default (simple projection), vit_film_tokenlearner (FiLM+TokenLearner), recurrent (with GRU)')
        parser.add_argument('--num_learned_tokens', dest='num_learned_tokens', type=int, default=8,
                            help='Number of learned tokens for TokenLearner (K)')
        parser.add_argument('--vision_emb_dim', dest='vision_emb_dim', type=int, default=128,
                            help='Vision embedding dimension after TokenLearner+FiLM readout')
        parser.add_argument('--gru_hidden_dim', dest='gru_hidden_dim', type=int, default=128,
                            help='GRU hidden state dimension for recurrent architectures')
        
        # NEW: Sampler mode selection
        parser.add_argument('--sampler_mode', dest='sampler_mode', type=str, default='map',
                            choices=['map', 'rect_regions', 'external'],
                            help='Sampler mode: map (distance transform), rect_regions (13 rectangles), external (spawn_goal_sampler)')
        
        # NEW: External sampler flag (deprecated, use --sampler_mode external)
        parser.add_argument('--use_external_sampler', dest='use_external_sampler', action='store_true', default=False,
                            help='DEPRECATED: Use --sampler_mode external instead')
        
        # NEW: Map-based sampler flag (deprecated, use --sampler_mode map)
        parser.add_argument('--use_map_sampler', dest='use_map_sampler', action='store_true', default=False,
                            help='DEPRECATED: Use --sampler_mode map instead')
        
        # NEW: Debug sampler flag (for noisy reset logs)
        parser.add_argument('--debug_sampler', dest='debug_sampler', action='store_true', default=False,
                            help='Print detailed reset logs to console (default: OFF)')
        
        # NEW: Distance-uniform sampling flag
        parser.add_argument('--distance_uniform', dest='distance_uniform', action='store_true', default=False,
                            help='Use distance-uniform goal sampling (sample distance bin first, then goal from bin) instead of spatial-uniform (default: OFF)')
        
        # NEW: Visualize sampler mode
        parser.add_argument('--visualize_sampler', dest='visualize_sampler', type=int, default=0,
                            help='Visualize sampler in Gazebo (cycles through N samples). Use with --viz_delay_sec. Set to 0 to disable (default: 0)')
        parser.add_argument('--viz_delay_sec', dest='viz_delay_sec', type=float, default=2.0,
                            help='Delay between visualization samples in seconds (default: 2.0)')
        
        # Tiny debug run flag (reduces timesteps only, does NOT bypass vision/ROS)
        parser.add_argument('--tiny_debug_run', dest='tiny_debug_run', action='store_true', default=False,
                            help='Fast debug mode: reduces timesteps to episode_max_steps=20, steps_per_iter=40, max_timesteps=400')
        
        # Reward type selection
        parser.add_argument('--reward_type', dest='reward_type', type=str, default='legacy',
                            choices=['legacy', 'fuzzy3', 'fuzzy3_v4'],
                            help='Reward function type: legacy (default), fuzzy3 (3-input Sugeno), or fuzzy3_v4 (anti-stall + robust clearance)')
        # Fixed case capture/replay for debugging
        parser.add_argument('--fixed_case_path', dest='fixed_case_path', type=str, default=None,
                            help='Path to fixed_case.json for replay mode (replays same start/goal)')
        
        # Distance curriculum arguments
        parser.add_argument('--curriculum_min_dist', dest='curriculum_min_dist', type=float, default=1.0,
                            help='Curriculum minimum goal distance in meters (default: 1.0)')
        parser.add_argument('--curriculum_max_dist', dest='curriculum_max_dist', type=float, default=8.0,
                            help='Curriculum maximum goal distance in meters (default: 8.0)')
        parser.add_argument('--curriculum_steps', dest='curriculum_steps', type=int, default=100000,
                            help='Number of training steps to ramp from min to max distance (default: 100000)')
        
        args = parser.parse_args()

        # Load config from baseline if specified
        if args.load_config:
                import yaml
                print(f"[arguments] Loading baseline config from: {args.load_config}", flush=True)
                with open(args.load_config, 'r') as f:
                        baseline_config = yaml.safe_load(f)
                
                # Override args with baseline config values
                if 'vision_backbone' in baseline_config:
                        args.vision_backbone = baseline_config['vision_backbone']
                if 'use_map_sampler' in baseline_config:
                        args.use_map_sampler = baseline_config['use_map_sampler']
                if 'distance_uniform' in baseline_config:
                        args.distance_uniform = baseline_config['distance_uniform']
                if 'reward_version' in baseline_config:
                        # Map reward_version back to reward_type if needed
                        pass  # We'll handle this in environment
                
                print(f"[arguments] Baseline config applied:", flush=True)
                print(f"  - vision_backbone: {args.vision_backbone}", flush=True)
                print(f"  - use_map_sampler: {args.use_map_sampler}", flush=True)
                print(f"  - distance_uniform: {args.distance_uniform}", flush=True)

        # Set default output_dir to repo_root/<runs_root> if not specified
        if args.output_dir is None:
                # Get repo root (3 levels up from src/ directory)
                src_dir = os.path.dirname(os.path.abspath(__file__))
                repo_root = os.path.abspath(os.path.join(src_dir, '..', '..', '..'))
                args.output_dir = os.path.join(repo_root, args.runs_root)
        
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
