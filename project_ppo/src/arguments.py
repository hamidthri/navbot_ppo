"""
	This file contains the arguments to parse at command line.
	File main.py will call get_args, which then the arguments
	will be returned.
"""
import argparse

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

	args = parser.parse_args()

	return args
