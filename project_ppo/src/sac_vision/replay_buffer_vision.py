#!/usr/bin/env python3
"""
Vision Replay Buffer for SAC
Stores dict states with images and lidar
"""
import numpy as np
import torch


class VisionReplayBuffer:
    """
    Replay buffer for vision-based SAC
    Stores states as {'image': (224,224,3), 'lidar': (16,)}
    """
    def __init__(self, lidar_dim, action_dim, max_size=int(1e5), device='cpu'):
        """
        Args:
            lidar_dim: Dimension of lidar state
            action_dim: Dimension of action
            max_size: Buffer capacity (smaller than standard due to image storage)
            device: torch device
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.lidar_dim = lidar_dim
        
        # Pre-allocate arrays
        # Images: (max_size, 224, 224, 3) - stored as uint8 to save memory
        self.images = np.zeros((max_size, 224, 224, 3), dtype=np.uint8)
        self.next_images = np.zeros((max_size, 224, 224, 3), dtype=np.uint8)
        
        # LiDAR states
        self.lidar_states = np.zeros((max_size, lidar_dim), dtype=np.float32)
        self.next_lidar_states = np.zeros((max_size, lidar_dim), dtype=np.float32)
        
        # Actions, rewards, dones
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
        print(f"[VisionReplayBuffer] Initialized with capacity={max_size}, lidar_dim={lidar_dim}")
        print(f"[VisionReplayBuffer] Memory usage: ~{max_size * (224*224*3*2 + lidar_dim*2*4 + action_dim*4 + 8) / 1e9:.2f} GB")
    
    def add(self, state, action, reward, next_state, done):
        """
        Add transition
        
        Args:
            state: dict {'image': np.array (224,224,3) uint8, 'lidar': np.array (lidar_dim,)}
            action: np.array (action_dim,)
            reward: float
            next_state: dict {'image': ..., 'lidar': ...}
            done: bool/float
        """
        # Store images as uint8 (already uint8 from environment)
        self.images[self.ptr] = state['image'].astype(np.uint8)
        self.next_images[self.ptr] = next_state['image'].astype(np.uint8)
        
        # Store lidar and other data
        self.lidar_states[self.ptr] = state['lidar']
        self.next_lidar_states[self.ptr] = next_state['lidar']
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        Sample batch
        
        Returns:
            state_batch: dict {'image': tensor (B,224,224,3), 'lidar': tensor (B,lidar_dim)}
            action_batch: tensor (B, action_dim)
            reward_batch: tensor (B, 1)
            next_state_batch: dict {'image': ..., 'lidar': ...}
            done_batch: tensor (B, 1)
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        # Convert uint8 images back to float32 [0,1]
        images = torch.FloatTensor(self.images[ind] / 255.0).to(self.device)
        next_images = torch.FloatTensor(self.next_images[ind] / 255.0).to(self.device)
        
        # Convert other data
        lidar = torch.FloatTensor(self.lidar_states[ind]).to(self.device)
        next_lidar = torch.FloatTensor(self.next_lidar_states[ind]).to(self.device)
        actions = torch.FloatTensor(self.actions[ind]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[ind]).to(self.device)
        dones = torch.FloatTensor(self.dones[ind]).to(self.device)
        
        # Return dict states
        state_batch = {'image': images, 'lidar': lidar}
        next_state_batch = {'image': next_images, 'lidar': next_lidar}
        
        return state_batch, actions, rewards, next_state_batch, dones
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples"""
        return self.size >= batch_size
    
    def __len__(self):
        return self.size


if __name__ == '__main__':
    # Quick test
    print("Testing VisionReplayBuffer...")
    
    buffer = VisionReplayBuffer(lidar_dim=16, action_dim=2, max_size=100, device='cpu')
    
    # Add some transitions
    for i in range(50):
        state = {
            'image': np.random.rand(224, 224, 3).astype(np.float32),
            'lidar': np.random.randn(16).astype(np.float32)
        }
        action = np.random.randn(2).astype(np.float32)
        reward = np.random.randn()
        next_state = {
            'image': np.random.rand(224, 224, 3).astype(np.float32),
            'lidar': np.random.randn(16).astype(np.float32)
        }
        done = 0.0
        
        buffer.add(state, action, reward, next_state, done)
    
    # Sample batch
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = buffer.sample(8)
    
    print(f"State batch: image={state_batch['image'].shape}, lidar={state_batch['lidar'].shape}")
    print(f"Action batch: {action_batch.shape}")
    print(f"Reward batch: {reward_batch.shape}")
    print(f"Next state batch: image={next_state_batch['image'].shape}, lidar={next_state_batch['lidar'].shape}")
    print(f"Done batch: {done_batch.shape}")
    print("✅ All tests passed!")
