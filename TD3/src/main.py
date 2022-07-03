#!/usr/bin/python -tt

import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
import rospy
from environment import Env
from torch.distributions import MultivariateNormal
from tensorboardX import SummaryWriter
import yaml
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env, time_step, eval_episodes=1):
    # eval_env = gym.make(env_name)
    # eval_env.seed(seed + 100)
    past_action = [0, 0]
    avg_reward = 0.
    t_s = 0
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done and t_s < time_step:
            action = policy.select_action(np.array(state))
            state, reward, done, arrive = env.step(action, past_action)
            avg_reward += reward
            t_s += 1
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def makepath(desired_path, isfile=False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


if __name__ == "__main__":
    logger = {
        'actor_losses': [],  # losses of actor network in current iteration
        'critic_losses': [],
        'current_Q': [],
        'target_Q': [],
        'Episode_Rewards': [],
    }
    logger_global = {
        'i_so_far': 0,  # iterations so far
        'actor_losses': [],  # losses of actor network in current iteration
        'critic_losses': [],
        'current_Q': [],
        'target_Q': [],
        'Episode_Rewards': [],
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")  # Policy name (TD3, DDPG or OurDDPG)
    # parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    # parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=6, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=6, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--lr", default=1e-4)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true", default=True)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    exp_id = 'simple_env_10'
    exp_path = f'./models/{exp_id}'
    makepath(exp_path)
    writer = SummaryWriter(log_dir=exp_path)

    ####

    conf = {
        'lr': args.lr,
        'gamma': args.discount,
        'save_freq': args.eval_freq,
    }
    with open(f'./models/{exp_id}' + '/config.yml', 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)
    # env = gym.make(args.env)
    is_training = True
    env = Env(is_training)
    rospy.init_node('ppo_stage_1.py')
    # Set seeds
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    state_dim = 16
    action_dim = 2
    max_action_l = 1
    min_action_l = 0
    max_action_a = 1
    min_action_a = -1
    max_action = 1
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "max_action_l": max_action_l,
        "max_action_a": max_action_a,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        print("----------load models------------")
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}", iteration=1)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    time_step = 5

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env, time_step)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    past_action = [0, 0]
    cov_var = torch.full(size=(action_dim,), fill_value=args.expl_noise).to(device)
    cov_mat = torch.diag(cov_var).to(device)
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1
        if t > 50000 and cov_mat[0][0] >= 0.1:
            cov_mat *= 0.995
        # Select action randomly or according to policy
        # if t < args.start_timesteps:
        # 	action = env.action_space.sample()
        # else:

        mean = torch.from_numpy(policy.select_action(np.array(state))).to(device)
        dist = MultivariateNormal(mean, cov_mat)
        action = dist.sample()

        action[0] = torch.clip(action[0], min_action_l, max_action_l)
        action[1] = torch.clip(action[1], min_action_a, max_action_a)
        action = action.cpu().data.numpy()
        # Perform action
        next_state, reward, done, arrive = env.step(action, past_action)
        done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        past_action = action
        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            actor_loss, critic_loss, current_Q, target_Q = policy.train(replay_buffer, args.batch_size)
            logger['actor_losses'].append(actor_loss.detach())
            logger['critic_losses'].append(critic_loss.detach())
            logger['current_Q'].append(current_Q.detach())
            logger['target_Q'].append(target_Q.detach())
        if arrive:
            result = 'Success'
        else:
            result = 'Fail'
        if arrive:
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward / episode_timesteps:.2f} Result: {result}")
            logger['Episode_Rewards'].append(episode_reward / episode_timesteps)

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if done or episode_timesteps >= time_step:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward / episode_timesteps:.2f} Result: {result}")
            logger['Episode_Rewards'].append(episode_reward / episode_timesteps)

            # Reset environment
            past_action = [0, 0]
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            if t>= args.start_timesteps:

                print(f"-------------------- Iteration #{logger_global['i_so_far'] + 1} --------------------", flush=True)
                avg_actor_loss = np.mean([losses.detach().cpu().mean() for losses in logger['actor_losses']])
                avg_critic_loss = np.mean([losses.detach().cpu().mean() for losses in logger['critic_losses']])
                print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
                print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
                print(f"------------------------------------------------------", flush=True)
                for k, v in logger.items():
                    if isinstance(v, list):
                        logger_global[k] += v
                all_steps = len(logger_global['actor_losses'])
                curr_steps = len(logger['actor_losses'])
                for i, loss in enumerate(logger['actor_losses']):
                    writer.add_scalar("Actor_loss/train", loss, all_steps - curr_steps + i)

                for i, loss in enumerate(logger['critic_losses']):
                    writer.add_scalar("Critic_loss/train", loss, all_steps - curr_steps + i)

                for i, loss in enumerate(logger['target_Q']):
                    writer.add_scalar("target_Q/train", loss, all_steps - curr_steps + i)
                for i, loss in enumerate(logger['current_Q']):
                    writer.add_scalar("current_Q/train", loss, all_steps - curr_steps + i)

                all_steps = len(logger_global['Episode_Rewards'])
                curr_steps = len(logger['Episode_Rewards'])
                for i, Reward in enumerate(logger['Episode_Rewards']):
                    writer.add_scalar("Episode_Rewards/train", Reward, all_steps - curr_steps + i)

                logger_global['i_so_far'] += 1
                logger['actor_losses'] = []
                logger['critic_losses'] = []
                logger['Episode_Rewards'] = []
                logger['current_Q'] = []
                logger['target_Q'] = []
            evaluations.append(eval_policy(policy, env, time_step))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}", logger_global['i_so_far'])
