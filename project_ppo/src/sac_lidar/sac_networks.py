#!/usr/bin/env python3
"""
Neural Network Architectures for SAC Algorithm

Implements:
- GaussianPolicy: Stochastic actor with squashed Gaussian distribution
- QNetwork: Twin Q-networks for value estimation
- Automatic entropy temperature tuning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# Initialize network weights
def weights_init_(m):
    """Initialize network weights using Xavier uniform initialization."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class GaussianPolicy(nn.Module):
    """
    Stochastic policy network for SAC.
    
    Outputs mean and log_std for a Gaussian distribution over actions.
    Uses tanh squashing to bound actions and applies reparameterization trick.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_space=None):
        """
        Initialize Gaussian policy.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer size
            action_space: Action space for scaling (optional)
        """
        super(GaussianPolicy, self).__init__()
        
        # Network architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Apply weight initialization
        self.apply(weights_init_)
        
        # Action bounds
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )
        
        # Epsilon for numerical stability
        self.epsilon = 1e-6
    
    def forward(self, state):
        """
        Forward pass through network.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            mean (torch.Tensor): Mean of Gaussian
            log_std (torch.Tensor): Log standard deviation
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample action from policy using reparameterization trick.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            action (torch.Tensor): Sampled action (tanh squashed)
            log_prob (torch.Tensor): Log probability of action
            mean (torch.Tensor): Mean action (deterministic, for evaluation)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from Normal distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Squash action with tanh
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability with change of variables formula
        log_prob = normal.log_prob(x_t)
        
        # Enforcing action bounds (see Appendix C of SAC paper)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Mean action for evaluation
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean
    
    def to(self, device):
        """Move network and action bounds to device."""
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class QNetwork(nn.Module):
    """
    Q-value network (critic) for SAC.
    
    Twin Q-networks to mitigate overestimation bias.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initialize Q-network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer size
        """
        super(QNetwork, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state, action):
        """
        Forward pass through both Q-networks.
        
        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor): Input action
            
        Returns:
            q1 (torch.Tensor): Q1 value
            q2 (torch.Tensor): Q2 value
        """
        xu = torch.cat([state, action], dim=1)
        
        # Q1 forward
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        q1 = self.fc3(x1)
        
        # Q2 forward
        x2 = F.relu(self.fc4(xu))
        x2 = F.relu(self.fc5(x2))
        q2 = self.fc6(x2)
        
        return q1, q2


class AlphaLearner(nn.Module):
    """
    Learnable entropy temperature for SAC.
    
    Automatically tunes the entropy coefficient to match a target entropy.
    """
    
    def __init__(self, target_entropy, log_alpha_init=0.0):
        """
        Initialize alpha learner.
        
        Args:
            target_entropy (float): Target entropy (typically -dim(action))
            log_alpha_init (float): Initial value for log(alpha)
        """
        super(AlphaLearner, self).__init__()
        
        self.target_entropy = target_entropy
        self.log_alpha = nn.Parameter(torch.tensor(log_alpha_init, dtype=torch.float32))
    
    def forward(self):
        """Return current alpha value."""
        return self.log_alpha.exp()
    
    def get_loss(self, log_prob):
        """
        Compute alpha loss for automatic tuning.
        
        Args:
            log_prob (torch.Tensor): Log probability of actions
            
        Returns:
            alpha_loss (torch.Tensor): Loss for alpha update
        """
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        return alpha_loss


if __name__ == "__main__":
    # Quick test
    print("Testing SAC Networks...")
    
    state_dim = 16
    action_dim = 2
    batch_size = 32
    
    # Test GaussianPolicy
    print("\n1. Testing GaussianPolicy...")
    policy = GaussianPolicy(state_dim, action_dim)
    states = torch.randn(batch_size, state_dim)
    actions, log_probs, mean_actions = policy.sample(states)
    
    print(f"  Action shape: {actions.shape}")
    print(f"  Log prob shape: {log_probs.shape}")
    print(f"  Mean action shape: {mean_actions.shape}")
    print(f"  Action range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    
    # Test QNetwork
    print("\n2. Testing QNetwork...")
    qnet = QNetwork(state_dim, action_dim)
    q1, q2 = qnet(states, actions)
    
    print(f"  Q1 shape: {q1.shape}")
    print(f"  Q2 shape: {q2.shape}")
    print(f"  Q1 value range: [{q1.min().item():.3f}, {q1.max().item():.3f}]")
    
    # Test AlphaLearner
    print("\n3. Testing AlphaLearner...")
    target_entropy = -action_dim
    alpha_learner = AlphaLearner(target_entropy)
    alpha = alpha_learner()
    alpha_loss = alpha_learner.get_loss(log_probs)
    
    print(f"  Initial alpha: {alpha.item():.6f}")
    print(f"  Alpha loss: {alpha_loss.item():.6f}")
    print(f"  Target entropy: {target_entropy}")
    print(f"  Current entropy: {-log_probs.mean().item():.6f}")
    
    # Test parameter count
    print("\n4. Parameter counts:")
    print(f"  Policy: {sum(p.numel() for p in policy.parameters())}")
    print(f"  QNetwork: {sum(p.numel() for p in qnet.parameters())}")
    print(f"  AlphaLearner: {sum(p.numel() for p in alpha_learner.parameters())}")
    
    print("\n✓ All SAC network tests passed!")
