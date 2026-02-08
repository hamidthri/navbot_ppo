#!/usr/bin/env python3
"""
Improved Vision SAC Networks with Deep Architecture
Features:
- Deeper MLP after fusion (3 layers instead of 1)
- Residual connections for better gradient flow
- LayerNorm for stability
- Optional dropout for regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from gate_fusion import get_fusion_module

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class LidarEncoder(nn.Module):
    """
    MLP encoder for lidar with LayerNorm
    """
    def __init__(self, lidar_dim, output_dim):
        super(LidarEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(lidar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        print(f"[LidarEncoder] {lidar_dim}D → {output_dim}D (with LayerNorm)")
    
    def forward(self, x):
        return self.encoder(x)


class ResidualBlock(nn.Module):
    """
    Residual block for deeper networks
    """
    def __init__(self, dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.fc1(x)))
        if self.dropout:
            out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        out = out + residual  # Residual connection
        out = F.relu(out)
        return out


class VisionActor(nn.Module):
    """
    Improved Actor with deeper network and residual connections
    """
    def __init__(self, lidar_dim, action_dim, hidden_dim, vision_backbone, 
                 action_space=None, fusion_type='concat', lidar_encoder_dim=256,
                 num_residual_blocks=2, dropout=0.0):
        super(VisionActor, self).__init__()
        
        self.vision_backbone = vision_backbone
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
        
        self.visual_dim = vision_backbone.output_dim
        
        # Lidar encoder
        self.lidar_encoder = LidarEncoder(lidar_dim, lidar_encoder_dim)
        
        # Fusion module
        self.fusion = get_fusion_module(
            fusion_type=fusion_type,
            visual_dim=self.visual_dim,
            lidar_dim=lidar_encoder_dim,
            output_dim=hidden_dim
        )
        
        # IMPROVED: Deeper policy network with residual blocks
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Add residual blocks for depth
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_blocks)
        ])
        
        # Final layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.action_space = action_space
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        print(f"[VisionActor] visual={self.visual_dim}D, lidar={lidar_dim}D→{lidar_encoder_dim}D, "
              f"hidden={hidden_dim}D, fusion={fusion_type}")
        print(f"  - Residual blocks: {num_residual_blocks}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Total depth: {3 + num_residual_blocks} layers after fusion")
        
    def forward(self, image, lidar_state):
        # Ensure correct image format
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        
        # Extract and normalize visual features
        with torch.no_grad():
            visual_features = self.vision_backbone(image)
            visual_features = F.layer_norm(visual_features, [self.visual_dim])
        
        # Encode lidar
        lidar_encoded = self.lidar_encoder(lidar_state)
        
        # Fuse
        fused = self.fusion(visual_features, lidar_encoded)
        
        # IMPROVED: Deeper policy network
        x = F.relu(self.norm1(self.fc1(fused)))
        if self.dropout:
            x = self.dropout(x)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final processing
        x = F.relu(self.norm2(self.fc2(x)))
        
        # Output
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(self, image, lidar_state):
        mean, log_std = self.forward(image, lidar_state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean)
        return action, log_prob, mean


class VisionCritic(nn.Module):
    """
    Improved Critic with deeper network and residual connections
    """
    def __init__(self, lidar_dim, action_dim, hidden_dim, vision_backbone, 
                 fusion_type='concat', lidar_encoder_dim=256,
                 num_residual_blocks=2, dropout=0.0):
        super(VisionCritic, self).__init__()
        
        self.vision_backbone = vision_backbone
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
        
        self.visual_dim = vision_backbone.output_dim
        
        # Lidar encoder
        self.lidar_encoder = LidarEncoder(lidar_dim, lidar_encoder_dim)
        
        # Fusion module
        self.fusion = get_fusion_module(
            fusion_type=fusion_type,
            visual_dim=self.visual_dim,
            lidar_dim=lidar_encoder_dim,
            output_dim=hidden_dim
        )
        
        # IMPROVED: Deeper Q-networks with residual blocks
        # Q1
        self.fc1_q1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.norm1_q1 = nn.LayerNorm(hidden_dim)
        
        self.residual_blocks_q1 = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_blocks)
        ])
        
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2_q1 = nn.LayerNorm(hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc1_q2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.norm1_q2 = nn.LayerNorm(hidden_dim)
        
        self.residual_blocks_q2 = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_blocks)
        ])
        
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2_q2 = nn.LayerNorm(hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        print(f"[VisionCritic] visual={self.visual_dim}D, lidar={lidar_dim}D→{lidar_encoder_dim}D, "
              f"action={action_dim}D, hidden={hidden_dim}D, fusion={fusion_type}")
        print(f"  - Residual blocks: {num_residual_blocks}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Total depth: {4 + num_residual_blocks} layers per Q-network")
    
    def forward(self, image, lidar_state, action):
        # Ensure correct image format
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        
        # Extract and normalize visual features
        with torch.no_grad():
            visual_features = self.vision_backbone(image)
            visual_features = F.layer_norm(visual_features, [self.visual_dim])
        
        # Encode lidar
        lidar_encoded = self.lidar_encoder(lidar_state)
        
        # Fuse
        fused = self.fusion(visual_features, lidar_encoded)
        
        # Concatenate with action
        x = torch.cat([fused, action], dim=-1)
        
        # Q1 - IMPROVED: Deeper network
        x1 = F.relu(self.norm1_q1(self.fc1_q1(x)))
        if self.dropout:
            x1 = self.dropout(x1)
        
        for block in self.residual_blocks_q1:
            x1 = block(x1)
        
        x1 = F.relu(self.norm2_q1(self.fc2_q1(x1)))
        q1 = self.fc3_q1(x1)
        
        # Q2 - IMPROVED: Deeper network
        x2 = F.relu(self.norm1_q2(self.fc1_q2(x)))
        if self.dropout:
            x2 = self.dropout(x2)
        
        for block in self.residual_blocks_q2:
            x2 = block(x2)
        
        x2 = F.relu(self.norm2_q2(self.fc2_q2(x2)))
        q2 = self.fc3_q2(x2)
        
        return q1, q2


if __name__ == '__main__':
    from vision_backbones import get_vision_backbone
    
    print("Testing Improved Vision Networks...")
    
    backbone = get_vision_backbone('resnet18', pretrained=False, output_dim=512)
    backbone.eval()
    
    lidar_dim = 16
    action_dim = 2
    hidden_dim = 256
    lidar_encoder_dim = 256
    batch_size = 4
    
    print("\n=== Testing with 2 residual blocks ===")
    actor = VisionActor(lidar_dim, action_dim, hidden_dim, backbone, 
                       fusion_type='concat', lidar_encoder_dim=lidar_encoder_dim,
                       num_residual_blocks=2, dropout=0.1)
    critic = VisionCritic(lidar_dim, action_dim, hidden_dim, backbone, 
                         fusion_type='concat', lidar_encoder_dim=lidar_encoder_dim,
                         num_residual_blocks=2, dropout=0.1)
    
    image = torch.rand(batch_size, 224, 224, 3)
    lidar = torch.randn(batch_size, lidar_dim)
    action = torch.randn(batch_size, action_dim)
    
    action_sample, log_prob, mean = actor.sample(image, lidar)
    print(f"Actor: action={action_sample.shape}, log_prob={log_prob.shape}")
    
    q1, q2 = critic(image, lidar, action)
    print(f"Critic: q1={q1.shape}, q2={q2.shape}")
    
    # Count parameters
    actor_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    critic_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"\nTrainable parameters:")
    print(f"  Actor:  {actor_params:,}")
    print(f"  Critic: {critic_params:,}")
    
    print("\n✅ All tests passed!")