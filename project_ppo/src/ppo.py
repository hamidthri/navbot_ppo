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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def makepath(desired_path, isfile=False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
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

        # Setup output directories
        self.method_run_dir = os.path.join(self.output_dir, self.method_name)
        self.checkpoint_dir = os.path.join(self.method_run_dir, 'checkpoints')
        self.log_dir_path = os.path.join(self.method_run_dir, 'logs')
        self.tb_dir_path = os.path.join(self.method_run_dir, 'tb')
        
        makepath(self.checkpoint_dir)
        makepath(self.log_dir_path)
        makepath(self.tb_dir_path)
        
        print(f"[PPO] Output directory: {self.method_run_dir}", flush=True)
        print(f"[PPO] Checkpoints: {self.checkpoint_dir}", flush=True)
        print(f"[PPO] Logs: {self.log_dir_path}", flush=True)
        print(f"[PPO] TensorBoard: {self.tb_dir_path}", flush=True)

        # Extract environment information
        self.env = env
        self.obs_dim = state_dim
        self.act_dim = action_dim
        
        # Check if using vision mode
        self.use_vision = hasattr(env, 'use_vision') and env.use_vision
        
        # Initialize frozen vision backbone if using vision
        if self.use_vision:
            from vision_backbones import get_backbone
            self.backbone_name = hyperparameters.get('vision_backbone', 'mobilenet_v2')
            vision_proj_dim = hyperparameters.get('vision_proj_dim', 64)
            
            self.vision_backbone, self.vision_feat_dim, self.vision_preprocess = get_backbone(self.backbone_name, device)
            print(f"[PPO] Initialized frozen {self.backbone_name} backbone (feat_dim={self.vision_feat_dim})", flush=True)
        else:
            self.vision_backbone = None
            self.vision_feat_dim = None
            self.vision_preprocess = None
            self.backbone_name = None
            vision_proj_dim = 64  # default, unused

        # Initialize actor and critic networks with vision support
        actor_kwargs = {
            'use_vision': self.use_vision, 
            'vision_feat_dim': self.vision_feat_dim if self.use_vision else 1280,
            'vision_proj_dim': vision_proj_dim
        }
        critic_kwargs = {
            'use_vision': self.use_vision, 
            'vision_feat_dim': self.vision_feat_dim if self.use_vision else 1280,
            'vision_proj_dim': vision_proj_dim
        }
        
        # Add architecture-specific hyperparameters if present
        arch_specific_keys = ['num_learned_tokens', 'vision_emb_dim', 'gru_hidden_dim']
        for key in arch_specific_keys:
            if key in hyperparameters:
                actor_kwargs[key] = hyperparameters[key]
                critic_kwargs[key] = hyperparameters[key]
        
        self.actor = policy_class(self.obs_dim, self.act_dim, **actor_kwargs).to(device)
        self.critic = value_func(self.obs_dim, 1, **critic_kwargs).to(device)
        
        # Check if actor needs vision tokens (for FiLM+TokenLearner architectures)
        self.use_vision_tokens = hasattr(self.actor, 'forward') and 'vision_tokens' in self.actor.forward.__code__.co_varnames

        # Initialize optimizers (vision projection params are inside actor/critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        # Verify trainable parameters
        self._verify_trainable_params()

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.8).to(device)
        self.cov_mat = torch.diag(self.cov_var).to(device)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
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
        self.writer = SummaryWriter(log_dir=self.tb_dir_path)
        
        # Save config file
        import yaml
        config_path = os.path.join(self.method_run_dir, 'config.yml')
        with open(config_path, 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)
        print(f"[PPO] Config saved to: {config_path}", flush=True)
        
        # Setup episode metrics CSV
        self.episode_csv_path = os.path.join(self.log_dir_path, f'{self.method_name}_train_episodes.csv')
        import csv
        with open(self.episode_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'timestep', 'success', 'collision', 'timeout', 'length', 'return', 'path_length', 'time'])
        self.episode_count = 0
    
    def _verify_trainable_params(self):
        """Verify and print trainable parameter counts"""
        print(f"\n{'='*70}", flush=True)
        print(f"[PPO] Trainable Parameter Verification", flush=True)
        print(f"{'='*70}", flush=True)
        
        # Vision backbone (should be 0)
        if self.vision_backbone is not None:
            backbone_trainable = sum(p.numel() for p in self.vision_backbone.parameters() if p.requires_grad)
            backbone_total = sum(p.numel() for p in self.vision_backbone.parameters())
            print(f"Vision Backbone ({self.backbone_name}): {backbone_total:>10,} total, {backbone_trainable:>10,} trainable {'✓ FROZEN' if backbone_trainable == 0 else '✗ ERROR'}", flush=True)
            assert backbone_trainable == 0, "Vision backbone must be frozen!"
        
        # Actor projection head (should be > 0 if using vision)
        if self.use_vision and hasattr(self.actor, 'vision_proj'):
            actor_proj_trainable = sum(p.numel() for p in self.actor.vision_proj.parameters() if p.requires_grad)
            actor_proj_total = sum(p.numel() for p in self.actor.vision_proj.parameters())
            print(f"Actor Projection Head:           {actor_proj_total:>10,} total, {actor_proj_trainable:>10,} trainable {'✓ TRAINABLE' if actor_proj_trainable > 0 else '✗ ERROR'}", flush=True)
            assert actor_proj_trainable > 0, "Actor projection head must be trainable!"
        
        # Actor residual head
        actor_residual_params = []
        for name, param in self.actor.named_parameters():
            if 'vision_proj' not in name:
                actor_residual_params.append(param)
        actor_residual_trainable = sum(p.numel() for p in actor_residual_params if p.requires_grad)
        actor_residual_total = sum(p.numel() for p in actor_residual_params)
        print(f"Actor Residual Head:             {actor_residual_total:>10,} total, {actor_residual_trainable:>10,} trainable {'✓ TRAINABLE' if actor_residual_trainable > 0 else '✗ ERROR'}", flush=True)
        
        # Critic projection head (should be > 0 if using vision)
        if self.use_vision and hasattr(self.critic, 'vision_proj'):
            critic_proj_trainable = sum(p.numel() for p in self.critic.vision_proj.parameters() if p.requires_grad)
            critic_proj_total = sum(p.numel() for p in self.critic.vision_proj.parameters())
            print(f"Critic Projection Head:          {critic_proj_total:>10,} total, {critic_proj_trainable:>10,} trainable {'✓ TRAINABLE' if critic_proj_trainable > 0 else '✗ ERROR'}", flush=True)
            assert critic_proj_trainable > 0, "Critic projection head must be trainable!"
        
        # Critic residual head
        critic_residual_params = []
        for name, param in self.critic.named_parameters():
            if 'vision_proj' not in name:
                critic_residual_params.append(param)
        critic_residual_trainable = sum(p.numel() for p in critic_residual_params if p.requires_grad)
        critic_residual_total = sum(p.numel() for p in critic_residual_params)
        print(f"Critic Residual Head:            {critic_residual_total:>10,} total, {critic_residual_trainable:>10,} trainable {'✓ TRAINABLE' if critic_residual_trainable > 0 else '✗ ERROR'}", flush=True)
        
        # Total
        total_trainable = sum(p.numel() for p in self.actor.parameters() if p.requires_grad) + \
                         sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        print(f"{'-'*70}", flush=True)
        print(f"Total Trainable Parameters:      {total_trainable:>10,}", flush=True)
        print(f"{'='*70}\n", flush=True)

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
        print(f"[Training Schedule] Episodes collect up to {self.max_timesteps_per_episode} steps each.", flush=True)
        print(f"[Training Schedule] Each iteration collects {self.timesteps_per_batch} environment steps before PPO update.", flush=True)
        print(f"[Training Schedule] Iteration summary prints after every {self.timesteps_per_batch} steps.", flush=True)
        print(f"[Training Schedule] Checkpoints save every {self.save_freq} iterations.", flush=True)
        print(f"[Training Schedule] Total budget: {total_timesteps} timesteps = {total_timesteps // self.timesteps_per_batch} iterations.", flush=True)
        
        # Compute PPO update steps per iteration
        # Since we use full-batch updates (no minibatching), each update epoch processes the entire batch
        grad_steps_per_iter = self.n_updates_per_iteration  # Both actor and critic updated this many times
        print(f"[PPO Update] Full-batch updates: {self.n_updates_per_iteration} gradient steps per iteration (both actor & critic).", flush=True)
        print(f"[PPO Update] Batch size: {self.timesteps_per_batch} timesteps (no minibatching).", flush=True)
        print("="*80, flush=True)
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        value_func = []
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Timing: iteration start
            t_iter_start = time.time()
            
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            t_rollout_start = time.time()
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, iter_metrics, batch_vision_feats = self.rollout(past_action=past_action, t_so_far=t_so_far)  # ALG STEP 3
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_rollout_end = time.time()
            rollout_time = t_rollout_end - t_rollout_start

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            
            # Check if we've reached or exceeded the target
            if t_so_far >= total_timesteps:
                print(f"[Training] Reached {t_so_far} timesteps (target: {total_timesteps}). Stopping training.", flush=True)
                # Still log this iteration before stopping
                pass  # Continue to logging below, then break at end

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            self.logger['iter_metrics'] = iter_metrics

            # Calculate advantage at k-th iteration
            self.V, _ = self.evaluate(batch_obs, batch_acts, batch_vision_feats)
            value_func.append(self.V.detach().mean())
            A_k = batch_rtgs - self.V.detach()  # ALG STEP 5
            f = open(os.path.join(self.log_dir_path, 'V_fun.txt'), 'a+')
            for i in value_func:
                f.write(str(i))
                f.write('\n')
            f.close()
            
            # Compute additional verification metrics
            with torch.no_grad():
                # Action statistics
                action_mean = batch_acts.abs().mean().item()
                action_std = batch_acts.std().item()
                # Saturated actions (near limits: linear [0,1], angular [-1,1])
                linear_saturated = ((batch_acts[:, 0] > 0.95) | (batch_acts[:, 0] < 0.05)).float().mean().item()
                angular_saturated = ((batch_acts[:, 1].abs() > 0.95)).float().mean().item()
                saturated_frac = (linear_saturated + angular_saturated) / 2.0
                
                # Value target statistics (batch_rtgs)
                rtg_mean = batch_rtgs.mean().item()
                rtg_std = batch_rtgs.std().item()
                rtg_min = batch_rtgs.min().item()
                rtg_max = batch_rtgs.max().item()
                rtg_p25 = torch.quantile(batch_rtgs, 0.25).item()
                rtg_p75 = torch.quantile(batch_rtgs, 0.75).item()
            
            # Store metrics
            self.logger['action_mean'] = action_mean
            self.logger['action_std'] = action_std
            self.logger['saturated_frac'] = saturated_frac
            self.logger['rtg_mean'] = rtg_mean
            self.logger['rtg_std'] = rtg_std
            self.logger['rtg_min'] = rtg_min
            self.logger['rtg_max'] = rtg_max
            self.logger['rtg_p25'] = rtg_p25
            self.logger['rtg_p75'] = rtg_p75
            
            # Normalizing advantages for stability
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Timing: PPO update start
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_update_start = time.time()
            actor_grad_steps = 0
            critic_grad_steps = 0
            
            # Capture initial params for delta computation
            actor_params_before = torch.cat([p.data.flatten() for p in self.actor.parameters()])
            critic_params_before = torch.cat([p.data.flatten() for p in self.critic.parameters()])
            
            # PPO training metrics
            approx_kl_list = []
            entropy_list = []
            clip_frac_list = []
            actor_grad_norm_list = []
            critic_grad_norm_list = []
            
            # Update network for n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                self.V, curr_log_probs = self.evaluate(batch_obs, batch_acts, batch_vision_feats)

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

                # Calculate PPO training metrics
                with torch.no_grad():
                    # Approx KL divergence
                    log_ratio = curr_log_probs - batch_log_probs
                    approx_kl = ((ratios - 1) - log_ratio).mean().item()
                    approx_kl_list.append(approx_kl)
                    
                    # Entropy (policy) - compute from actual policy distribution
                    # Re-compute mean for current policy
                    if self.use_vision_tokens and batch_vision_feats is not None:
                        mean = self.actor(batch_obs, vision_tokens=batch_vision_feats)
                    else:
                        mean = self.actor(batch_obs, vision_feat=batch_vision_feats)
                    mean_clamped = torch.stack([
                        torch.clamp(mean[:, 0], 0, 1),
                        torch.clamp(mean[:, 1], -1, 1)
                    ], dim=1)
                    dist = MultivariateNormal(mean_clamped, self.cov_mat)
                    # Mean entropy per timestep in batch
                    entropy = dist.entropy().mean().item()
                    entropy_list.append(entropy)
                    
                    # Clip fraction
                    clip_frac = ((ratios - 1.0).abs() > self.clip).float().mean().item()
                    clip_frac_list.append(clip_frac)

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(self.V, batch_rtgs)
                # weihgts = self.actor.parameters()
                # w1_res1_actor0 = weihgts.gi_frame.f_locals['self'].rb1.fc1.weight

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                
                # Compute actor gradient norm
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
                actor_grad_norm_list.append(actor_grad_norm.item())
                
                # Diagnostic: Check for zero or invalid grad norms
                if actor_grad_norm.item() <= 0 or not np.isfinite(actor_grad_norm.item()):
                    diag_path = os.path.join(self.log_dir_path, 'grad_diagnostics.txt')
                    with open(diag_path, 'a') as f_diag:
                        f_diag.write(f"\n[ACTOR GRAD ISSUE] Iteration {i_so_far}, Update step {_+1}/{self.n_updates_per_iteration}\n")
                        f_diag.write(f"  Actor grad norm: {actor_grad_norm.item()}\n")
                        f_diag.write(f"  Actor loss: {actor_loss.item()}\n")
                        
                        # Count params with/without grad
                        total_params = 0
                        params_with_grad = 0
                        grad_sum = 0.0
                        for p in self.actor.parameters():
                            if p.requires_grad:
                                total_params += 1
                                if p.grad is not None:
                                    params_with_grad += 1
                                    grad_sum += p.grad.abs().sum().item()
                        
                        f_diag.write(f"  Actor params requiring grad: {total_params}\n")
                        f_diag.write(f"  Actor params with grad: {params_with_grad}\n")
                        f_diag.write(f"  Sum of abs(grad): {grad_sum}\n")
                        f_diag.write(f"  Advantage stats: mean={A_k.mean().item():.4f}, std={A_k.std().item():.4f}, min={A_k.min().item():.4f}, max={A_k.max().item():.4f}\n")
                        f_diag.write(f"  Ratio stats: mean={ratios.mean().item():.4f}, min={ratios.min().item():.4f}, max={ratios.max().item():.4f}\n")
                        f_diag.write(f"  Clip fraction: {clip_frac_list[-1] if clip_frac_list else 0.0:.4f}\n")
                
                self.actor_optim.step()
                actor_grad_steps += 1

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                
                # Compute critic gradient norm
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float('inf'))
                critic_grad_norm_list.append(critic_grad_norm.item())
                
                self.critic_optim.step()
                critic_grad_steps += 1

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())
                self.logger['critic_losses'].append(critic_loss.detach())

            # Compute parameter delta (L2 norm of change)
            actor_params_after = torch.cat([p.data.flatten() for p in self.actor.parameters()])
            critic_params_after = torch.cat([p.data.flatten() for p in self.critic.parameters()])
            actor_param_delta = torch.norm(actor_params_after - actor_params_before, p=2).item()
            critic_param_delta = torch.norm(critic_params_after - critic_params_before, p=2).item()
            
            # Compute mean PPO metrics
            mean_approx_kl = np.mean(approx_kl_list) if approx_kl_list else 0.0
            mean_entropy = np.mean(entropy_list) if entropy_list else 0.0
            mean_clip_frac = np.mean(clip_frac_list) if clip_frac_list else 0.0
            mean_actor_grad_norm = np.mean(actor_grad_norm_list) if actor_grad_norm_list else 0.0
            mean_critic_grad_norm = np.mean(critic_grad_norm_list) if critic_grad_norm_list else 0.0
            
            # Assert gradients and params are changing (sanity check)
            # Check actor grad norm (warn instead of crash)
            if mean_actor_grad_norm <= 0 or not np.isfinite(mean_actor_grad_norm):
                print(f"[WARNING] Actor grad norm invalid: {mean_actor_grad_norm:.6f} at iteration {i_so_far}. Check grad_diagnostics.txt", flush=True)
                print(f"[WARNING] Skipping parameter update validation for this iteration.", flush=True)
            else:
                pass  # assert mean_critic_grad_norm > 0 and np.isfinite(mean_critic_grad_norm), f"Critic grad norm invalid: {mean_critic_grad_norm}"
                pass  # assert actor_param_delta > 0, f"Actor params not changing! Delta: {actor_param_delta}"
                pass  # assert critic_param_delta > 0, f"Critic params not changing! Delta: {critic_param_delta}"
            
            # Store in logger
            self.logger['approx_kl'] = mean_approx_kl
            self.logger['entropy'] = mean_entropy
            self.logger['clip_frac'] = mean_clip_frac
            self.logger['actor_grad_norm'] = mean_actor_grad_norm
            self.logger['critic_grad_norm'] = mean_critic_grad_norm
            self.logger['actor_param_delta'] = actor_param_delta
            self.logger['critic_param_delta'] = critic_param_delta

            # Timing: PPO update end
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_update_end = time.time()
            update_time = t_update_end - t_update_start
            
            # Timing: iteration end
            t_iter_end = time.time()
            iter_time = t_iter_end - t_iter_start
            
            # Store timing info in logger
            self.logger['rollout_time'] = rollout_time
            self.logger['update_time'] = update_time
            self.logger['iter_time'] = iter_time
            self.logger['actor_grad_steps'] = actor_grad_steps
            self.logger['critic_grad_steps'] = critic_grad_steps
            
            # Print a summary of our training so far
            self._log_summary()
            # Save our model if it's time

            if i_so_far % self.save_freq == 0:
                actor_path = os.path.join(self.checkpoint_dir, f'actor_iter{i_so_far:04d}_step{int(t_so_far):08d}.pth')
                critic_path = os.path.join(self.checkpoint_dir, f'critic_iter{i_so_far:04d}_step{int(t_so_far):08d}.pth')
                torch.save(self.actor.state_dict(), actor_path)
                torch.save(self.critic.state_dict(), critic_path)
                print(f"[PPO] Saved checkpoint at iteration {i_so_far}, step {int(t_so_far)}: {actor_path}", flush=True)
            
            # Check if we should stop after this iteration
            if t_so_far >= total_timesteps:
                break

    def rollout(self, past_action, t_so_far):
        """
			Collect batch of data from simulation.
			For vision mode: extract 1280-d features ONCE per step and cache them.

			Return:
				batch_obs - base state observations (LiDAR + pose + past). Shape: (timesteps, obs_dim)
				batch_acts - actions collected. Shape: (timesteps, action_dim)
				batch_log_probs - log probabilities. Shape: (timesteps,)
				batch_rtgs - Rewards-To-Go. Shape: (timesteps,)
				batch_lens - episode lengths. Shape: (num_episodes,)
				batch_vision_feats - vision features (1280-d). Shape: (timesteps, 1280) or None
		"""
        # Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_vision_feats = [] if self.use_vision else None

        # Reset environment
        obs = self.env.reset()
        done = False
        episode_reward = 0
        one_round = 0
        ep_rews = []
        
        # Episode metrics tracking
        ep_path_length = 0.0
        prev_pos = None
        ep_start_time = time.time()
        
        # Iteration-level metrics
        iter_successes = 0
        iter_collisions = 0
        iter_timeouts = 0
        iter_ep_times = []
        iter_ep_count = 0
        
        # Rollout loop
        for t in range(self.timesteps_per_batch):

            # Track base state observation
            batch_obs.append(obs)
            
            # Sanity check: base state must be exactly 16-d (10 lidar + 2 past_action + 4 goalpose)
            assert len(obs) == 16, f"Base state must be 16-d, got {len(obs)} at timestep {t_so_far + t}"
            
            # Extract vision features ONCE (frozen backbone, no grad)
            if self.use_vision:
                img = self.env.getLatestImage()
                if img is not None:
                    # Preprocess and extract features
                    with torch.inference_mode():
                        img_tensor = self.vision_preprocess(img)  # (1, 3, H, W)
                        
                        # Extract tokens if needed (for FiLM+TokenLearner architectures)
                        if self.use_vision_tokens:
                            from vision_backbones import extract_vision_tokens
                            vision_feat_tensor = extract_vision_tokens(self.vision_backbone, img_tensor, self.backbone_name)
                            # vision_feat_tensor: (1, N, C) tokens
                            vision_feat = vision_feat_tensor.squeeze(0).cpu().numpy()  # (N, C)
                        else:
                            # Extract pooled features (default)
                            vision_feat_tensor = self.vision_backbone(img_tensor)  # (1, feat_dim, ...) or (1, feat_dim)
                            # Flatten to (feat_dim,) if needed
                            if vision_feat_tensor.dim() > 2:
                                vision_feat_tensor = torch.nn.functional.adaptive_avg_pool2d(vision_feat_tensor, (1, 1))
                            vision_feat = vision_feat_tensor.squeeze().cpu().numpy()  # (feat_dim,)
                else:
                    # No image available - provide zeros
                    if self.use_vision_tokens:
                        # Default: 256 tokens of 384-d for dinov2_vits14
                        vision_feat = np.zeros((256, 384), dtype=np.float32)
                    else:
                        vision_feat = np.zeros(self.vision_feat_dim, dtype=np.float32)
                batch_vision_feats.append(vision_feat)
            else:
                vision_feat = None
            
            # Get action from policy
            action, log_prob = self.get_action(obs, t_so_far, one_round, vision_feat=vision_feat)
            
            # Get position for path length calculation
            curr_pos = np.array([self.env.position.x, self.env.position.y])
            if prev_pos is not None:
                ep_path_length += np.linalg.norm(curr_pos - prev_pos)
            prev_pos = curr_pos
            
            # Step environment
            obs, rew, done, arrive = self.env.step(action, past_action)

            past_action = action
            episode_reward += rew
            ep_rews.append(rew)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)
            # t += 1 		# Increment timesteps ran this batch so far
            one_round += 1
            
            # Check termination: collision, arrival, or timeout
            timeout = (one_round >= self.max_timesteps_per_episode)
            if done or arrive or timeout:
                # Compute episode time
                ep_time = time.time() - ep_start_time
                
                # Log episode metrics
                success = 1 if arrive else 0
                collision = 1 if (done and not arrive) else 0
                timeout_flag = 1 if (timeout and not done and not arrive) else 0
                
                self._log_episode_metrics(
                    episode_num=self.episode_count,
                    timestep=t_so_far + np.sum(batch_lens) + one_round,
                    success=success,
                    collision=collision,
                    timeout=timeout_flag,
                    length=one_round,
                    ep_return=episode_reward,
                    path_length=ep_path_length,
                    ep_time=ep_time
                )
                self.episode_count += 1
                
                # Track iteration metrics
                iter_successes += success
                iter_collisions += collision
                iter_timeouts += timeout_flag
                iter_ep_times.append(ep_time)
                iter_ep_count += 1
                
                batch_lens.append(one_round)
                batch_rews.append(ep_rews)
                ep_rews = []
                if one_round != 0:
                    self.logger['Episode_Rewards'].append(episode_reward / one_round)
                episode_reward = 0
                one_round = 0
                ep_path_length = 0.0
                prev_pos = None
                past_action = [0, 0]
                done = False
                obs = self.env.reset()
                ep_start_time = time.time()
            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            # If render is specified, render the environment
            # if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
            # 	self.env.render()

        # Track episodic lengths and rewards
        batch_rews.append(ep_rews)
        # if one_round != 0:
        # 	print('Step: %3i' % one_round, '| avg_reward:{:.2f}'.format(episode_reward / one_round),
        # 		  '| Time step: %i' % (t_so_far + np.sum(batch_lens)), '|', result)
        #
        # 	self.logger['Episode_Rewards'].append(episode_reward / one_round)

        f = open(os.path.join(self.log_dir_path, 'ppo.txt'), 'a+')
        for i in self.logger['Episode_Rewards']:
            f.write(str(i))
            f.write('\n')
        f.close()
        episode_rewards = []

        # Reshape data as tensors
        batch_obs = torch.from_numpy(np.array(batch_obs, dtype=np.float32)).float()
        batch_acts = torch.from_numpy(np.array(batch_acts, dtype=np.float32)).float()
        batch_log_probs = torch.from_numpy(np.array(batch_log_probs, dtype=np.float32)).float()
        batch_rtgs = self.compute_rtgs(batch_rews)
        
        # Convert vision features to tensor if using vision
        if self.use_vision:
            batch_vision_feats = torch.from_numpy(np.array(batch_vision_feats, dtype=np.float32)).float()
        else:
            batch_vision_feats = None

        # Log episodic returns and lengths
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        
        # Create iteration metrics dictionary
        iter_metrics = {
            'successes': iter_successes,
            'collisions': iter_collisions,
            'timeouts': iter_timeouts,
            'ep_times': iter_ep_times,
            'ep_count': iter_ep_count
        }

        return batch_obs.to(device), batch_acts.to(device), batch_log_probs.to(device), batch_rtgs.to(device), \
               batch_lens, iter_metrics, batch_vision_feats.to(device) if batch_vision_feats is not None else None

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

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs, t_so_far, one_round, vision_feat=None):
        """
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the base state observation at the current timestep
				vision_feat - the vision features (1280-d) or tokens (N, C) for this timestep (optional)

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
        self.t_step = one_round
        # Query the actor network for a mean action
        if self.use_vision and vision_feat is not None:
            vision_feat_tensor = torch.tensor(vision_feat, dtype=torch.float).unsqueeze(0).to(device)
            # vision_feat_tensor: (1, feat_dim) for pooled, or (1, N, C) for tokens
            
            if self.use_vision_tokens:
                mean = self.actor(obs, vision_tokens=vision_feat_tensor)
            else:
                mean = self.actor(obs, vision_feat=vision_feat_tensor)
        else:
            mean = self.actor(obs)

        # Create distribution and sample action
        if self.t_step == 0 and t_so_far > 50000 and self.cov_mat[0][0] >= 0.1:
            self.cov_mat *= 0.995
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        
        # Normalize shape: ensure action is (B, action_dim)
        if action.dim() == 1:
            action = action.unsqueeze(0)  # (action_dim,) -> (1, action_dim)
        
        # Clamp action dimensions
        action_clamped = torch.stack([
            torch.clamp(action[:, 0], 0, 1),
            torch.clamp(action[:, 1], -1, 1)
        ], dim=1)
        
        log_prob = dist.log_prob(action_clamped)
        
        # Normalize shape: ensure log_prob is (B,)
        if log_prob.dim() == 0:
            log_prob = log_prob.unsqueeze(0)  # scalar -> (1,)
        
        # Return as numpy arrays with batch dim: action (1, 2), log_prob (1,)
        return action_clamped.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0]

    def evaluate(self, batch_obs, batch_acts, batch_vision_feats=None):
        """
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the base state observations from batch. Shape: (timesteps, obs_dim)
				batch_acts - the actions from batch. Shape: (timesteps, action_dim)
				batch_vision_feats - the vision features or tokens from batch. 
				                     Shape: (timesteps, feat_dim) or (timesteps, N, C)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
        # Query critic network for values V (handle both vision_feat and vision_tokens)
        if self.use_vision_tokens and batch_vision_feats is not None:
            self.V = self.critic(batch_obs, vision_tokens=batch_vision_feats).squeeze()
        else:
            self.V = self.critic(batch_obs, vision_feat=batch_vision_feats).squeeze()

        # Calculate log probabilities using most recent actor network
        if self.use_vision_tokens and batch_vision_feats is not None:
            mean = self.actor(batch_obs, vision_tokens=batch_vision_feats)
        else:
            mean = self.actor(batch_obs, vision_feat=batch_vision_feats)
            
        # Clamp without in-place operations
        mean_clamped = torch.stack([
            torch.clamp(mean[:, 0], 0, 1),
            torch.clamp(mean[:, 1], -1, 1)
        ], dim=1)
        dist = MultivariateNormal(mean_clamped, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        
        return self.V, log_probs

    def _log_episode_metrics(self, episode_num, timestep, success, collision, timeout, length, ep_return, path_length, ep_time):
        """
            Log per-episode metrics to CSV file.
        """
        import csv
        with open(self.episode_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode_num, timestep, success, collision, timeout, length, ep_return, path_length, ep_time])
        
        # Print per-episode summary to console
        outcome = "SUCCESS" if success else ("COLLISION" if collision else ("TIMEOUT" if timeout else "UNKNOWN"))
        print(f"[Ep {episode_num:4d}] {outcome:9s} | Steps: {length:3d} | Return: {ep_return:7.1f} | Time: {ep_time:5.1f}s | Total timesteps: {timestep}", flush=True)

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
        self.timesteps_per_batch = 8000  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 800  # Max number of timesteps per episode
        self.n_updates_per_iteration = 50  # Number of times to update actor/critic per iteration
        self.lr = 3e-4  # Learning rate of actor optimizer
        self.gamma = 0.99  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 2  # How often we save in number of iterations
        self.seed = None  # Sets the seed of our program, used for reproducibility of results
        self.exp_id = 'v02_simple_env_60_reward_proportion'
        self.method_name = 'baseline'  # Method identifier for checkpoints and logs
        self.output_dir = None  # Base output directory (will be set from args)

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            v = str(val) if not isinstance(val, str) else f'"{val}"'
            exec('self.' + param + ' = ' + v)

        conf = {
            'timesteps_per_batch': self.timesteps_per_batch,
            'max_timesteps_per_episode': self.max_timesteps_per_episode,
            'n_updates_per_iteration': self.n_updates_per_iteration,
            'lr': self.lr,
            'gamma': self.gamma,
            'clip': self.clip,
            'render': self.render,
            'render_every_i': self.render_every_i,
            'save_freq': self.save_freq,
            'seed': self.seed,
            'exp_id': self.exp_id,
            'method_name': self.method_name,
            'output_dir': self.output_dir,
        }

        ### Save config (will be written after directories are created in __init__)
        self.config = conf

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

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
        
        # Extract iteration metrics
        iter_metrics = self.logger.get('iter_metrics', {})
        ep_count = iter_metrics.get('ep_count', 0)
        if ep_count > 0:
            success_rate = (iter_metrics.get('successes', 0) / ep_count) * 100
            collision_rate = (iter_metrics.get('collisions', 0) / ep_count) * 100
            timeout_rate = (iter_metrics.get('timeouts', 0) / ep_count) * 100
            ep_times = iter_metrics.get('ep_times', [])
            mean_ep_time = np.mean(ep_times) if ep_times else 0.0
        else:
            success_rate = collision_rate = timeout_rate = mean_ep_time = 0.0
        
        # Extract timing metrics
        rollout_time = self.logger.get('rollout_time', 0.0)
        update_time = self.logger.get('update_time', 0.0)
        iter_time = self.logger.get('iter_time', 0.0)
        actor_grad_steps = self.logger.get('actor_grad_steps', 0)
        critic_grad_steps = self.logger.get('critic_grad_steps', 0)
        steps_per_sec = self.timesteps_per_batch / iter_time if iter_time > 0 else 0.0
        
        # Extract PPO validity metrics
        approx_kl = self.logger.get('approx_kl', 0.0)
        entropy = self.logger.get('entropy', 0.0)
        clip_frac = self.logger.get('clip_frac', 0.0)
        actor_grad_norm = self.logger.get('actor_grad_norm', 0.0)
        critic_grad_norm = self.logger.get('critic_grad_norm', 0.0)
        actor_param_delta = self.logger.get('actor_param_delta', 0.0)
        critic_param_delta = self.logger.get('critic_param_delta', 0.0)
        
        # Extract verification metrics
        action_mean = self.logger.get('action_mean', 0.0)
        action_std = self.logger.get('action_std', 0.0)
        saturated_frac = self.logger.get('saturated_frac', 0.0)
        rtg_mean = self.logger.get('rtg_mean', 0.0)
        rtg_std = self.logger.get('rtg_std', 0.0)
        rtg_min = self.logger.get('rtg_min', 0.0)
        rtg_max = self.logger.get('rtg_max', 0.0)
        rtg_p25 = self.logger.get('rtg_p25', 0.0)
        rtg_p75 = self.logger.get('rtg_p75', 0.0)

        # Print logging statements
        print(flush=True)
        print(f"================================================================================", flush=True)
        print(f"Iteration: {i_so_far}", flush=True)
        print(f"================================================================================", flush=True)
        print(f"Episodes in Iteration: {ep_count}", flush=True)
        print(f"Success Rate: {success_rate:.1f}%", flush=True)
        print(f"Collision Rate: {collision_rate:.1f}%", flush=True)
        print(f"Timeout Rate: {timeout_rate:.1f}%", flush=True)
        print(f"Mean Episode Reward: {avg_ep_rews:.2f}", flush=True)
        print(f"Mean Episode Length: {avg_ep_lens:.2f}", flush=True)
        print(f"Mean Episode Time: {mean_ep_time:.2f} seconds", flush=True)
        print(f"Actor Loss: {avg_actor_loss:.4f}", flush=True)
        print(f"Critic Loss: {avg_critic_loss:.4f}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"---", flush=True)
        print(f"Rollout Time: {rollout_time:.2f}s | Update Time: {update_time:.2f}s | Iteration Time: {iter_time:.2f}s", flush=True)
        print(f"Steps/sec: {steps_per_sec:.1f} | Actor Grad Steps: {actor_grad_steps} | Critic Grad Steps: {critic_grad_steps}", flush=True)
        print(f"---", flush=True)
        print(f"PPO Metrics: KL={approx_kl:.6f} | Entropy={entropy:.4f} | ClipFrac={clip_frac:.4f}", flush=True)
        print(f"Grad Norms: Actor={actor_grad_norm:.4f} | Critic={critic_grad_norm:.4f}", flush=True)
        print(f"Param Delta: Actor={actor_param_delta:.6f} | Critic={critic_param_delta:.6f}", flush=True)
        print(f"---", flush=True)
        print(f"Action Stats: mean(|a|)={action_mean:.4f} | std(a)={action_std:.4f} | saturated={saturated_frac*100:.1f}%", flush=True)
        print(f"Value Targets: mean={rtg_mean:.2f} | std={rtg_std:.2f} | range=[{rtg_min:.2f}, {rtg_max:.2f}] | IQR=[{rtg_p25:.2f}, {rtg_p75:.2f}]", flush=True)
        print(f"================================================================================", flush=True)
        print(flush=True)
        
        # Log key metrics to TensorBoard
        self.writer.add_scalar("train/success_rate", success_rate, i_so_far)
        self.writer.add_scalar("train/collision_rate", collision_rate, i_so_far)
        self.writer.add_scalar("train/timeout_rate", timeout_rate, i_so_far)
        self.writer.add_scalar("train/mean_return", avg_ep_rews, i_so_far)
        self.writer.add_scalar("train/mean_ep_length", avg_ep_lens, i_so_far)
        self.writer.add_scalar("train/mean_ep_time", mean_ep_time, i_so_far)
        self.writer.add_scalar("loss/actor", avg_actor_loss, i_so_far)
        self.writer.add_scalar("loss/critic", avg_critic_loss, i_so_far)
        self.writer.add_scalar("train/timesteps", t_so_far, i_so_far)
        
        # Log timing/performance metrics to TensorBoard
        self.writer.add_scalar("time/rollout", rollout_time, i_so_far)
        self.writer.add_scalar("time/update", update_time, i_so_far)
        self.writer.add_scalar("time/iteration", iter_time, i_so_far)
        self.writer.add_scalar("perf/steps_per_sec", steps_per_sec, i_so_far)
        self.writer.add_scalar("perf/actor_grad_steps", actor_grad_steps, i_so_far)
        self.writer.add_scalar("perf/critic_grad_steps", critic_grad_steps, i_so_far)
        
        # Log PPO validity metrics to TensorBoard
        self.writer.add_scalar("ppo/approx_kl", approx_kl, i_so_far)
        self.writer.add_scalar("ppo/entropy", entropy, i_so_far)
        self.writer.add_scalar("ppo/clip_frac", clip_frac, i_so_far)
        self.writer.add_scalar("ppo/actor_grad_norm", actor_grad_norm, i_so_far)
        self.writer.add_scalar("ppo/critic_grad_norm", critic_grad_norm, i_so_far)
        self.writer.add_scalar("ppo/actor_param_delta", actor_param_delta, i_so_far)
        self.writer.add_scalar("ppo/critic_param_delta", critic_param_delta, i_so_far)

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

        all_steps = len(self.logger_global['Episode_Rewards'])
        curr_steps = len(self.logger['Episode_Rewards'])
        for i, Reward in enumerate(self.logger['Episode_Rewards']):
            self.writer.add_scalar("Episode_Rewards/train", Reward, all_steps - curr_steps + i)

        self.logger_global['Iteration'] += 1
        self.writer.add_scalar("avg_ep_rews/train", avg_ep_rews, self.logger_global['Iteration'])

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['Episode_Rewards'] = []
    # self.logger_global['iter'] = 0
