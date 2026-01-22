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

	args = parser.parse_args()
	
	# Set default output_dir to repo_root/runs if not specified
	if args.output_dir is None:
		# Get repo root (3 levels up from src/ directory)
		src_dir = os.path.dirname(os.path.abspath(__file__))
		repo_root = os.path.abspath(os.path.join(src_dir, '..', '..', '..'))
		args.output_dir = os.path.join(repo_root, 'runs')
	
	return args

