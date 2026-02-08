#!/usr/bin/env python3
"""
Vision-enabled SAC Agent with Stability Fixes
FIXED: Added gradient clipping, reward normalization, Q-value clipping
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from sac_networks_vision import VisionActor, VisionCritic
from replay_buffer_vision import VisionReplayBuffer


class AlphaLearner(torch.nn.Module):
    """Learns entropy temperature alpha"""
    def __init__(self, target_entropy):
        super(AlphaLearner, self).__init__()
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        self.target_entropy = target_entropy
        
    def forward(self):
        return self.log_alpha.exp()
    
    def get_loss(self, log_prob):
        return -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()


class VisionSAC:
    """
    SAC agent for vision-based navigation with stability fixes
    """
    def __init__(
        self,
        lidar_dim,
        action_dim,
        vision_backbone,
        device='cpu',
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        buffer_size=int(1e6),
        batch_size=256,
        action_space=None,
        fusion_type='concat',
        lidar_encoder_dim=256,
        # STABILITY PARAMETERS (NEW)
        gradient_clip_norm=1.0,
        reward_scale=100.0,  # Divide rewards by this
        use_reward_normalization=False,
        q_value_clip=None  # None or (min, max) tuple
    ):
        """
        Initialize vision SAC agent with stability fixes
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.lidar_dim = lidar_dim
        self.fusion_type = fusion_type
        
        # STABILITY PARAMETERS
        self.gradient_clip_norm = gradient_clip_norm
        self.reward_scale = reward_scale
        self.use_reward_normalization = use_reward_normalization
        self.q_value_clip = q_value_clip
        
        # Reward normalization statistics
        if self.use_reward_normalization:
            self.reward_mean = 0.0
            self.reward_var = 1.0
            self.reward_count = 0
        
        # Initialize networks
        self.policy = VisionActor(
            lidar_dim, action_dim, hidden_dim, vision_backbone, 
            action_space, fusion_type, lidar_encoder_dim
        ).to(device)
        
        self.critic = VisionCritic(
            lidar_dim, action_dim, hidden_dim, vision_backbone, 
            fusion_type, lidar_encoder_dim
        ).to(device)
        
        self.critic_target = VisionCritic(
            lidar_dim, action_dim, hidden_dim, vision_backbone, 
            fusion_type, lidar_encoder_dim
        ).to(device)
        
        # Hard update target
        self.hard_update(self.critic_target, self.critic)
        
        # Optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        
        # Entropy temperature
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.alpha_learner = AlphaLearner(self.target_entropy).to(device)
            self.alpha_optimizer = Adam([self.alpha_learner.log_alpha], lr=lr_alpha)
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.memory = VisionReplayBuffer(lidar_dim, action_dim, buffer_size, device)
        
        self.total_updates = 0
        
        print(f"[VisionSAC] Initialized: lidar_dim={lidar_dim}, action_dim={action_dim}, fusion={fusion_type}")
        print(f"[VisionSAC] Stability fixes:")
        print(f"  - Gradient clipping: {gradient_clip_norm}")
        print(f"  - Reward scale: /{reward_scale}")
        print(f"  - Reward normalization: {use_reward_normalization}")
        print(f"  - Q-value clipping: {q_value_clip}")
        print(f"[VisionSAC] Lidar encoder: {lidar_dim}D → {lidar_encoder_dim}D")
        print(f"[VisionSAC] Backbone frozen: {not any(p.requires_grad for p in vision_backbone.parameters())}")
    
    def normalize_reward(self, reward):
        """
        Normalize reward using running statistics
        """
        if not self.use_reward_normalization:
            # Just scale
            return reward / self.reward_scale
        
        # Update running statistics
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var = self.reward_var + delta * delta2
        
        # Normalize
        std = np.sqrt(self.reward_var / self.reward_count) if self.reward_count > 1 else 1.0
        normalized = (reward - self.reward_mean) / (std + 1e-8)
        
        # Clip to prevent outliers
        return np.clip(normalized, -10.0, 10.0)
    
    def select_action(self, state, evaluate=False):
        """Select action from policy"""
        # Convert to tensors and normalize image to [0,1]
        image = torch.FloatTensor(state['image'] / 255.0).unsqueeze(0).to(self.device)
        lidar = torch.FloatTensor(state['lidar']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(image, lidar)
            else:
                action, _, _ = self.policy.sample(image, lidar)
        
        return action.cpu().numpy()[0]
    
    def update(self):
        """Perform one SAC update step with stability fixes"""
        if not self.memory.is_ready(self.batch_size):
            return {}
        
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        # Extract components
        image_batch = state_batch['image']
        lidar_batch = state_batch['lidar']
        next_image_batch = next_state_batch['image']
        next_lidar_batch = next_state_batch['lidar']
        
        # Update critic
        with torch.no_grad():
            next_action_batch, next_log_prob_batch, _ = self.policy.sample(next_image_batch, next_lidar_batch)
            q1_next_target, q2_next_target = self.critic_target(next_image_batch, next_lidar_batch, next_action_batch)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            
            if self.automatic_entropy_tuning:
                alpha = self.alpha_learner().detach()
            else:
                alpha = self.alpha
            
            min_q_next_target = min_q_next_target - alpha * next_log_prob_batch
            
            # FIX: Clip Q-values to prevent explosion
            if self.q_value_clip is not None:
                min_q_next_target = torch.clamp(min_q_next_target, self.q_value_clip[0], self.q_value_clip[1])
            
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target
        
        q1, q2 = self.critic(image_batch, lidar_batch, action_batch)
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # FIX: Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.gradient_clip_norm)
        self.critic_optimizer.step()
        
        # Update actor
        action_batch_new, log_prob_batch, _ = self.policy.sample(image_batch, lidar_batch)
        q1_new, q2_new = self.critic(image_batch, lidar_batch, action_batch_new)
        min_q_new = torch.min(q1_new, q2_new)
        
        if self.automatic_entropy_tuning:
            alpha = self.alpha_learner().detach()
        else:
            alpha = self.alpha
        
        actor_loss = (alpha * log_prob_batch - min_q_new).mean()
        
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        # FIX: Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.gradient_clip_norm)
        self.policy_optimizer.step()
        
        # Update alpha
        alpha_loss = torch.tensor(0.0)
        if self.automatic_entropy_tuning:
            alpha_loss = self.alpha_learner.get_loss(log_prob_batch)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.critic_target, self.critic)
        
        self.total_updates += 1
        
        # Return statistics
        if self.automatic_entropy_tuning:
            alpha_value = self.alpha_learner().item()
        else:
            alpha_value = self.alpha
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else 0.0,
            'alpha': alpha_value,
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
        }
    
    def soft_update(self, target, source):
        """Soft update: target = tau * source + (1 - tau) * target"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def hard_update(self, target, source):
        """Hard update: target = source"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self, filepath):
        """Save model checkpoints"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_learner_state_dict': self.alpha_learner.state_dict() if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'total_updates': self.total_updates,
        }
        
        if self.use_reward_normalization:
            checkpoint['reward_mean'] = self.reward_mean
            checkpoint['reward_var'] = self.reward_var
            checkpoint['reward_count'] = self.reward_count
        
        torch.save(checkpoint, filepath)
    
    def load(self, filepath):
        """Load model checkpoints"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if self.automatic_entropy_tuning and checkpoint['alpha_learner_state_dict'] is not None:
            self.alpha_learner.load_state_dict(checkpoint['alpha_learner_state_dict'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.total_updates = checkpoint['total_updates']
        
        if self.use_reward_normalization and 'reward_mean' in checkpoint:
            self.reward_mean = checkpoint['reward_mean']
            self.reward_var = checkpoint['reward_var']
            self.reward_count = checkpoint['reward_count']