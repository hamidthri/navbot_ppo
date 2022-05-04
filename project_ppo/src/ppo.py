"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gym
import time

import numpy as np
import time
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import os, glob
import torch.nn.functional as F

saver_dir = './models/'
if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)

record_dir = 'record'
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, value_func, env, state_dim, action_dim, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		# assert(type(env.observation_space) == gym.spaces.Box)
		# assert(type(env.action_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		self.obs_dim = state_dim
		self.act_dim = action_dim

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim).to(device)                                              # ALG STEP 1
		self.critic = value_func(self.obs_dim, 1).to(device)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5).to(device)
		self.cov_mat = torch.diag(self.cov_var).to(device)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'critic_losses': [],
			'Episode_Rewards': [],
		}

		self.logger_global = {
			'delta_t': time.time(),
			't_so_far': 0,  # timesteps so far
			'i_so_far': 0,  # iterations so far
			'batch_lens': [],  # episodic lengths in batch
			'batch_rews': [],  # episodic returns in batch
			'actor_losses': [],  # losses of actor network in current iteration
			'critic_losses': [],
			'Episode_Rewards': [],
			'Iteration': 0,
		}
		self.writer = SummaryWriter(log_dir=self.log_dir)



	def learn(self, total_timesteps, past_action):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		value_func = []
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout(past_action=past_action, t_so_far=t_so_far)                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			self.V, _ = self.evaluate(batch_obs, batch_acts)
			value_func.append(self.V.detach().mean())
			A_k = batch_rtgs - self.V.detach()                                                                       # ALG STEP 5
			f = open(record_dir + '/V_fun' + '.txt', 'a+')
			for i in value_func:
				f.write(str(i))
				f.write('\n')
			f.close()
			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				self.V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(self.V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())
				self.logger['critic_losses'].append(critic_loss.detach())

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time

			if i_so_far % self.save_freq == 0:
				epoch = i_so_far//self.save_freq
				torch.save(self.actor.state_dict(), f'models/{self.exp_id}/ppo_actor_{epoch}.pth')
				torch.save(self.critic.state_dict(), f'models/{self.exp_id}/ppo_critic_{epoch}.pth')

	def rollout(self, past_action, t_so_far):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		# episode_rewards = []
		t = 0 # Keeps track of how many timesteps we've run so far this batch
		# Keep simulating until we've run more than or equal to specified timesteps per batch
		# while t < self.timesteps_per_batch:
		# 	ep_rews = []   # rewards collected per episode
		#
		# 	# Reset the environment. sNote that obs is short for observation.
		# 	obs = self.env.reset()
		# 	episode_reward = 0
		# 	done = False
		# 	one_round = 0
		# 	# Run an episode for a maximum of max_timesteps_per_episode timesteps
		# 	# for ep_t in range(self.max_timesteps_per_episode):
		# 	while True:
		# 		# If render is specified, render the environment
		# 		# if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
		# 		# 	self.env.render()
		# 		t += 1 		# Increment timesteps ran this batch so far
		#
		# 		# Track observations in this batch
		# 		batch_obs.append(obs)
		# 		# Calculate action and make a step in the env.
		# 		# Note that rew is short for reward.
		# 		action, log_prob = self.get_action(obs, t_so_far)
		# 		obs, rew, done, arrive = self.env.step(action, past_action)
		# 		past_action = action
		# 		episode_reward += rew
		# 		ep_t += 1
		# 		one_round += 1
		# 		# Track recent reward, action, and action log probability
		# 		ep_rews.append(rew)
		# 		batch_acts.append(action)
		# 		batch_log_probs.append(log_prob)
		#
		# 		# If the environment tells us the episode is terminated, break
		# 		if arrive:
		# 			result = 'Success'
		# 		else:
		# 			result = 'Fail'
		# 		if arrive:
		# 			print('Step: %3i' % one_round, '| avg_reward:{:.2f}'.format(episode_reward/one_round),  '|', result)
		# 			episode_rewards.append([episode_reward])
		# 			episode_reward = 0
		# 			one_round = 0
		# 			if t >= self.timesteps_per_batch:
		# 				break
		# 		if done or one_round >= 500:
		# 			print('Step: %3i' % one_round, '| avg_reward:{:.2f}'.format(episode_reward/one_round), '|', result)
		# 			break
		# Reset the environment. sNote that obs is short for observation.
		obs = self.env.reset()
		done = False
		episode_reward = 0
		one_round = 0
		ep_rews = []
		# while t < self.timesteps_per_batch:
		for t in range(self.timesteps_per_batch):

			# Track observations in this batch
			batch_obs.append(obs)
			# Calculate action and make a step in the env.
			# Note that rew is short for reward.
			action, log_prob = self.get_action(obs, t_so_far, one_round)
			# action[1] = action[1] * 2 / 3
			obs, rew, done, arrive = self.env.step(action, past_action)
			past_action = action
			episode_reward += rew
			# Track recent reward, action, and action log probability
			ep_rews.append(rew)
			batch_acts.append(action)
			batch_log_probs.append(log_prob)
			# t += 1 		# Increment timesteps ran this batch so far
			one_round += 1
			if done or one_round >= self.max_timesteps_per_episode:
				batch_lens.append(one_round)
				batch_rews.append(ep_rews)
				ep_rews = []
				if one_round != 0:
					print('Step: %3i' % one_round, '| avg_reward:{:.2f}'.format(episode_reward / one_round),
						  '| Time step: %i' % (t_so_far + np.sum(batch_lens)), '|', result)
					self.logger['Episode_Rewards'].append(episode_reward / one_round)
				episode_reward = 0
				one_round = 0
				done = False
				obs = self.env.reset()
			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			# If render is specified, render the environment
			# if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
			# 	self.env.render()
			# If the environment tells us the episode is terminated, break
			if arrive:
				result = 'Success'
			else:
				result = 'Fail'
			if arrive:
				batch_rews.append(ep_rews)
				ep_rews = []
				batch_lens.append(one_round)
				if one_round != 0:
					print('Step: %3i' % one_round, '| avg_reward:{:.2f}'.format(episode_reward/one_round), '| Time step: %i' % (t_so_far + np.sum(batch_lens)), '|', result)
					self.logger['Episode_Rewards'].append(episode_reward/one_round)
				episode_reward = 0
				one_round = 0

		# Track episodic lengths and rewards
		batch_rews.append(ep_rews)
		if one_round != 0:
			self.logger['Episode_Rewards'].append(episode_reward / one_round)
		f = open(record_dir + '/ppo' + '.txt', 'a+')
		for i in self.logger['Episode_Rewards']:
			f.write(str(i))
			f.write('\n')
		f.close()
		episode_rewards = []

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_rtgs = self.compute_rtgs(batch_rews)

		# ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs.to(device), batch_acts.to(device), batch_log_probs.to(device), batch_rtgs.to(device), batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs

	def get_action(self, obs, t_so_far, one_round):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		self.t_step = one_round
		# Query the actor network for a mean action
		mean = self.actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		if self.t_step % 50 == 0 and t_so_far > 30000 and self.cov_mat[0][0] >= 0.1:
			self.cov_mat *= 0.95
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()
		# action1 = action
		# action[0] = F.sigmoid(action[0])
		# action[1] = F.tanh(action[1])
		# action[1] = 1.5 * action[1]
		action[0] = torch.clip(action[0], 0, 1)
		action[1] = torch.clip(action[1], -1, 1)
		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().cpu().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		self.V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		# mean[0] = F.sigmoid(mean[0]).detach()
		# mean[1] = F.tanh(mean[1]).detach()
		# mean[1] = 1.5 * mean[1]
		mean[0] = torch.clip(mean[0], 0, 1).detach()
		mean[1] = torch.clip(mean[1], -1, 1).detach()
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return self.V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 8000               # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 800          # Max number of timesteps per episode
		self.n_updates_per_iteration = 50               # Number of times to update actor/critic per iteration
		self.lr = 3e-4                                # Learning rate of actor optimizer
		self.gamma = 0.99                            # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
		self.log_dir = 'log'
		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 2                            # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results
		self.exp_id = 'V07'

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			v = str(val) if not isinstance(val, str) else f'"{val}"'
			exec('self.' + param + ' = ' + v)

		conf = {
			'timesteps_per_batch': self.timesteps_per_batch,
			'max_timesteps_per_episode': self.max_timesteps_per_episode,
			'n_updates_per_iteration' :self.n_updates_per_iteration,
			'lr': self.lr,
			'gamma': self.gamma,
			'clip': self.clip,
			'log_dir': self.log_dir,
			'render': self.render,
			'render_every_i': self.render_every_i,
			'save_freq': self.save_freq,
			'seed': self.seed,
			'exp_id': self.exp_id,

		}



		### create model folder
		exp_path = f'models/{self.exp_id}'
		makepath(exp_path)
		####
		import yaml
		with open(f'models/{self.exp_id}'+'/config.yml', 'w') as outfile:
			yaml.dump(conf, outfile, default_flow_style=False)

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))
		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.detach().cpu().mean() for losses in self.logger['actor_losses']])
		avg_critic_loss = np.mean([losses.detach().cpu().mean() for losses in self.logger['critic_losses']])
		# print(avg_actor_loss)
		# print(avg_critic_loss)
		# Round decimal places for more aesthetic logging messages
		# avg_ep_lens = str(round(avg_ep_lens, 2))
		# avg_ep_rews = str(round(avg_ep_rews, 2))
		# avg_actor_loss = str(round(avg_actor_loss, 5))
		# avg_critic_loss = str(round(avg_critic_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
		print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		## take all logging data

		for k, v in self.logger.items():
			if isinstance(v, list):
				self.logger_global[k] += v

		all_steps = len(self.logger_global['actor_losses'])
		curr_steps = len(self.logger['actor_losses'])
		for i, loss in enumerate(self.logger['actor_losses']):
			self.writer.add_scalar("Actor_loss/train", loss, all_steps - curr_steps + i)

		for i, loss in enumerate(self.logger['critic_losses']):
			self.writer.add_scalar("Critic_loss/train", loss, all_steps - curr_steps + i)
		# self.logger['Episode_Rewards'].append(avg_ep_rews)
		self.logger_global['Iteration'] += 1
		self.writer.add_scalar("avg_ep_rews/train", avg_ep_rews, self.logger_global['Iteration'])

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
		self.logger['critic_losses'] = []
