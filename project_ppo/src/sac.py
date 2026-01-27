#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) Algorithm

Implementation of SAC for continuous control in robot navigation.
Features:
- Twin Q-networks to mitigate overestimation
- Automatic entropy temperature tuning
- Soft target updates
- Off-policy learning with replay buffer
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from sac_networks import GaussianPolicy, QNetwork, AlphaLearner
from replay_buffer import ReplayBuffer


class SAC:
    """
    Soft Actor-Critic algorithm for continuous control.
    
    Combines:
    - Maximum entropy RL (encourages exploration)
    - Off-policy learning (sample efficient)
    - Actor-critic architecture
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        device='cpu',
        # Network hyperparameters
        hidden_dim=256,
        # Learning rates
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        # SAC hyperparameters
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        # Replay buffer
        buffer_size=int(1e6),
        batch_size=256,
        # Training
        action_space=None
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            device (str): Device to use ('cpu' or 'cuda')
            hidden_dim (int): Hidden layer size
            lr_actor (float): Learning rate for policy
            lr_critic (float): Learning rate for Q-networks
            lr_alpha (float): Learning rate for entropy temperature
            gamma (float): Discount factor
            tau (float): Soft update coefficient
            alpha (float): Initial entropy temperature (if not auto-tuning)
            automatic_entropy_tuning (bool): Whether to auto-tune alpha
            buffer_size (int): Replay buffer capacity
            batch_size (int): Batch size for updates
            action_space: Action space for scaling
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Initialize networks
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim, action_space).to(device)
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Copy parameters to target network
        self.hard_update(self.critic_target, self.critic)
        
        # Optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        
        # Entropy temperature
        if self.automatic_entropy_tuning:
            # Target entropy = -dim(action space)
            self.target_entropy = -action_dim
            self.alpha_learner = AlphaLearner(self.target_entropy).to(device)
            self.alpha_optimizer = Adam([self.alpha_learner.log_alpha], lr=lr_alpha)
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.memory = ReplayBuffer(state_dim, action_dim, buffer_size, device)
        
        # Training statistics
        self.total_updates = 0
    
    def select_action(self, state, evaluate=False):
        """
        Select action from policy.
        
        Args:
            state (np.ndarray): Current state
            evaluate (bool): If True, use mean action (deterministic)
            
        Returns:
            action (np.ndarray): Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
        
        return action.cpu().numpy()[0]
    
    def update(self):
        """
        Perform one SAC update step.
        
        Returns:
            dict: Training statistics (losses, Q-values, etc.)
        """
        if not self.memory.is_ready(self.batch_size):
            return {}
        
        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.memory.sample(self.batch_size)
        
        # ========== Update critic ==========
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_action_batch, next_log_prob_batch, _ = self.policy.sample(next_state_batch)
            
            # Compute target Q-values using target networks
            q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_action_batch)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            
            # Add entropy term
            if self.automatic_entropy_tuning:
                alpha = self.alpha_learner().detach()
            else:
                alpha = self.alpha
            
            min_q_next_target = min_q_next_target - alpha * next_log_prob_batch
            
            # Compute target: r + γ(1 - done) * min_Q_target(s', a')
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target
        
        # Current Q-values
        q1, q2 = self.critic(state_batch, action_batch)
        
        # Critic loss: MSE between current Q and target Q
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ========== Update actor ==========
        # Sample actions from current policy
        action_batch_new, log_prob_batch, _ = self.policy.sample(state_batch)
        
        # Q-values for new actions
        q1_new, q2_new = self.critic(state_batch, action_batch_new)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Actor loss: maximize Q - α*log_prob (equivalent to minimizing negative)
        if self.automatic_entropy_tuning:
            alpha = self.alpha_learner().detach()
        else:
            alpha = self.alpha
        
        actor_loss = (alpha * log_prob_batch - min_q_new).mean()
        
        # Optimize actor
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()
        
        # ========== Update entropy temperature ==========
        alpha_loss = torch.tensor(0.0)
        if self.automatic_entropy_tuning:
            alpha_loss = self.alpha_learner.get_loss(log_prob_batch)
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # ========== Soft update target networks ==========
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        self.total_updates += 1
        
        # Return training statistics
        stats = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.automatic_entropy_tuning else 0.0,
            'alpha': alpha if not self.automatic_entropy_tuning else self.alpha_learner().item(),
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'log_prob_mean': log_prob_batch.mean().item(),
            'entropy': -log_prob_batch.mean().item()
        }
        
        return stats
    
    def soft_update(self, target, source, tau):
        """
        Soft update target network parameters: θ_target = τ*θ_source + (1-τ)*θ_target
        
        Args:
            target: Target network
            source: Source network
            tau (float): Interpolation parameter
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    
    def hard_update(self, target, source):
        """
        Hard update target network parameters: θ_target = θ_source
        
        Args:
            target: Target network
            source: Source network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self, filename):
        """
        Save model parameters.
        
        Args:
            filename (str): Path to save file
        """
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_updates': self.total_updates
        }
        
        if self.automatic_entropy_tuning:
            checkpoint['alpha_learner_state_dict'] = self.alpha_learner.state_dict()
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, filename)
        print(f"[SAC] Model saved to {filename}")
    
    def load(self, filename):
        """
        Load model parameters.
        
        Args:
            filename (str): Path to load file
        """
        if not os.path.exists(filename):
            print(f"[SAC] Warning: {filename} not found, starting from scratch")
            return
        
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_updates = checkpoint['total_updates']
        
        if self.automatic_entropy_tuning and 'alpha_learner_state_dict' in checkpoint:
            self.alpha_learner.load_state_dict(checkpoint['alpha_learner_state_dict'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        print(f"[SAC] Model loaded from {filename} (updates: {self.total_updates})")


if __name__ == "__main__":
    # Quick test
    print("Testing SAC Algorithm...")
    
    state_dim = 16
    action_dim = 2
    
    # Initialize SAC
    sac = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        device='cpu',
        batch_size=32,
        buffer_size=1000
    )
    
    print(f"\n1. SAC initialized:")
    print(f"   Policy parameters: {sum(p.numel() for p in sac.policy.parameters())}")
    print(f"   Critic parameters: {sum(p.numel() for p in sac.critic.parameters())}")
    print(f"   Device: {sac.device}")
    
    # Test action selection
    print(f"\n2. Testing action selection...")
    state = np.random.randn(state_dim)
    action = sac.select_action(state, evaluate=False)
    print(f"   Stochastic action: {action}")
    
    action_det = sac.select_action(state, evaluate=True)
    print(f"   Deterministic action: {action_det}")
    
    # Fill replay buffer
    print(f"\n3. Filling replay buffer...")
    for i in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        
        sac.memory.add(state, action, reward, next_state, done)
    
    print(f"   Buffer size: {len(sac.memory)}")
    
    # Test update
    print(f"\n4. Testing SAC update...")
    stats = sac.update()
    
    if stats:
        print(f"   Critic loss: {stats['critic_loss']:.4f}")
        print(f"   Actor loss: {stats['actor_loss']:.4f}")
        print(f"   Alpha: {stats['alpha']:.4f}")
        print(f"   Q1 mean: {stats['q1_mean']:.4f}")
        print(f"   Entropy: {stats['entropy']:.4f}")
    
    # Test save/load
    print(f"\n5. Testing save/load...")
    sac.save('/tmp/sac_test.pth')
    sac.load('/tmp/sac_test.pth')
    
    print("\n✓ All SAC tests passed!")
