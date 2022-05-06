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

from environment import Env
import os, glob


state_dim = 26
action_dim = 2
action_linear_max = 0.25  # m/s
action_angular_max = 1  # rad/s
is_training = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(env, hyperparameters, actor_model, critic_model):
    """
        Trains the model.

        Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training

        Return:
            None
    """
    print(f"Start Training ... ", flush=True)

    # Create a model for PPO.
    agent = PPO(policy_class=NetActor, value_func=NetCritic, env=env, state_dim=state_dim, action_dim=action_dim,
                **hyperparameters)
    past_action = np.array([0., 0.])

    # Tries to load in an existing actor/critic model to continue training on
    critic_path = sorted(glob.glob(f'models/{agent.exp_id}/ppo_critic_*.pth'))
    actor_path  = sorted(glob.glob(f'models/{agent.exp_id}/ppo_actor_*.pth'))
    if critic_path != []:
        print(f"Loading in {actor_path[-1]} and {critic_path[-1]}...", flush=True)
        agent.actor.load_state_dict(torch.load(actor_path[-1]))
        agent.critic.load_state_dict(torch.load(critic_path[-1]))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '':  # Don't train from scratch if user accidentally forgets actor/critic model
        print(
            f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    agent.learn(total_timesteps=2000000, past_action=past_action)
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
    env = Env(is_training)
    # agent = DDPG(env, state_dim, action_dim)


    print('State Dimensions: ' + str(state_dim))
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
        'timesteps_per_batch': 8000,
        'max_timesteps_per_episode': 800,
        'gamma': 0.99,
        'n_updates_per_iteration': 50,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10,
        'log_dir': '/home/hamid/repos/planner/catkin_ws/src/project_ppo/src/summary',
        'exp_id': "V07"
    }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym and have both continuous
    # observation and action spaces.
    # Train or test, depending on the mode specified
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model,
              critic_model=args.critic_model)
        ### env.logger_global
    else:
        test(env=env, actor_model=args.actor_model)


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    main(args)
