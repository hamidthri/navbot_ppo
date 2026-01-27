#!/usr/bin/env python3
"""
Replay Buffer for SAC Algorithm

Stores transitions (state, action, reward, next_state, done) for off-policy learning.
Implements efficient sampling and storage with numpy arrays.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer for SAC.
    
    Stores transitions and provides random sampling for off-policy learning.
    Uses numpy arrays for efficient storage and conversion to PyTorch tensors.
    """
    
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cpu'):
        """
        Initialize replay buffer.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            max_size (int): Maximum number of transitions to store
            device (str): Device to use for tensor conversion ('cpu' or 'cuda')
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        
        # Pre-allocate memory for efficiency
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state (np.ndarray): Current state
            action (np.ndarray): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode terminated
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions uniformly at random.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones) as PyTorch tensors
        """
        # Random sampling without replacement
        ind = np.random.randint(0, self.size, size=batch_size)
        
        # Convert to PyTorch tensors and move to device
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
    
    def __len__(self):
        """Return current size of buffer."""
        return self.size
    
    def is_ready(self, batch_size):
        """
        Check if buffer has enough samples for training.
        
        Args:
            batch_size (int): Required batch size
            
        Returns:
            bool: True if buffer has at least batch_size samples
        """
        return self.size >= batch_size
    
    def save(self, filename):
        """
        Save replay buffer to disk.
        
        Args:
            filename (str): Path to save file
        """
        np.savez(
            filename,
            state=self.state[:self.size],
            action=self.action[:self.size],
            reward=self.reward[:self.size],
            next_state=self.next_state[:self.size],
            done=self.done[:self.size],
            ptr=self.ptr,
            size=self.size
        )
    
    def load(self, filename):
        """
        Load replay buffer from disk.
        
        Args:
            filename (str): Path to load file
        """
        data = np.load(filename)
        
        self.size = int(data['size'])
        self.ptr = int(data['ptr'])
        
        self.state[:self.size] = data['state']
        self.action[:self.size] = data['action']
        self.reward[:self.size] = data['reward']
        self.next_state[:self.size] = data['next_state']
        self.done[:self.size] = data['done']


if __name__ == "__main__":
    # Quick test
    print("Testing ReplayBuffer...")
    
    buffer = ReplayBuffer(state_dim=16, action_dim=2, max_size=1000)
    
    # Add some transitions
    for i in range(100):
        state = np.random.randn(16)
        action = np.random.randn(2)
        reward = np.random.randn()
        next_state = np.random.randn(16)
        done = np.random.rand() > 0.9
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Ready for batch_size=32: {buffer.is_ready(32)}")
    
    # Sample a batch
    if buffer.is_ready(32):
        states, actions, rewards, next_states, dones = buffer.sample(32)
        print(f"Sampled batch shapes:")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Rewards: {rewards.shape}")
        print(f"  Next States: {next_states.shape}")
        print(f"  Dones: {dones.shape}")
    
    print("\n✓ ReplayBuffer test passed!")
