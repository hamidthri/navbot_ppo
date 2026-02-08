#!/usr/bin/env python3
"""
Gate Fusion Mechanisms for Multimodal Learning
Implements FiLM (Feature-wise Linear Modulation) and other fusion strategies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for fusing visual and LiDAR features.
    
    FiLM learns to modulate one modality (visual) with another (LiDAR) using
    affine transformations: gamma * visual_features + beta
    
    Reference: "FiLM: Visual Reasoning with a General Conditioning Layer"
    """
    def __init__(self, visual_dim, lidar_dim, output_dim=None, verbose=True):
        super(FiLMFusion, self).__init__()
        self.visual_dim = visual_dim
        self.lidar_dim = lidar_dim
        self.output_dim = output_dim if output_dim is not None else visual_dim
        
        # FiLM generator: produces gamma and beta from LiDAR features
        self.film_generator = nn.Sequential(
            nn.Linear(lidar_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * visual_dim)  # 2x for gamma and beta
        )
        
        # Optional output projection
        if output_dim is not None and output_dim != visual_dim:
            self.output_proj = nn.Linear(visual_dim, output_dim)
        else:
            self.output_proj = None
            
        if verbose:
            print(f"[FiLMFusion] visual_dim={visual_dim}, lidar_dim={lidar_dim}, output_dim={self.output_dim}")
        
    def forward(self, visual_features, lidar_features):
        """
        Args:
            visual_features: Visual features (B, visual_dim)
            lidar_features: LiDAR features (B, lidar_dim)
            
        Returns:
            fused_features: Fused features (B, output_dim)
        """
        # Generate FiLM parameters
        film_params = self.film_generator(lidar_features)  # (B, 2 * visual_dim)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # Each: (B, visual_dim)
        
        # Apply FiLM modulation
        modulated = gamma * visual_features + beta  # (B, visual_dim)
        
        # Optional output projection
        if self.output_proj is not None:
            modulated = self.output_proj(modulated)  # (B, output_dim)
            
        return modulated


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion with MLP
    """
    def __init__(self, visual_dim, lidar_dim, output_dim, verbose=True):
        super(ConcatFusion, self).__init__()
        self.visual_dim = visual_dim
        self.lidar_dim = lidar_dim
        self.output_dim = output_dim
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(visual_dim + lidar_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        if verbose:
            print(f"[ConcatFusion] visual_dim={visual_dim}, lidar_dim={lidar_dim}, output_dim={output_dim}")
        
    def forward(self, visual_features, lidar_features):
        """
        Args:
            visual_features: Visual features (B, visual_dim)
            lidar_features: LiDAR features (B, lidar_dim)
            
        Returns:
            fused_features: Fused features (B, output_dim)
        """
        concatenated = torch.cat([visual_features, lidar_features], dim=-1)
        fused = self.fusion_mlp(concatenated)
        return fused


class AttentionFusion(nn.Module):
    """
    Attention-based fusion: LiDAR attends to visual features
    """
    def __init__(self, visual_dim, lidar_dim, output_dim, verbose=True):
        super(AttentionFusion, self).__init__()
        self.visual_dim = visual_dim
        self.lidar_dim = lidar_dim
        self.output_dim = output_dim
        
        # Attention mechanism
        self.query_proj = nn.Linear(lidar_dim, visual_dim)
        self.key_proj = nn.Linear(visual_dim, visual_dim)
        self.value_proj = nn.Linear(visual_dim, visual_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(visual_dim + lidar_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        if verbose:
            print(f"[AttentionFusion] visual_dim={visual_dim}, lidar_dim={lidar_dim}, output_dim={output_dim}")
        
    def forward(self, visual_features, lidar_features):
        """
        Args:
            visual_features: Visual features (B, visual_dim)
            lidar_features: LiDAR features (B, lidar_dim)
            
        Returns:
            fused_features: Fused features (B, output_dim)
        """
        # Compute attention
        query = self.query_proj(lidar_features)  # (B, visual_dim)
        key = self.key_proj(visual_features)  # (B, visual_dim)
        value = self.value_proj(visual_features)  # (B, visual_dim)
        
        # Scaled dot-product attention (simplified for 1D features)
        attention_weights = F.softmax(query * key, dim=-1)  # (B, visual_dim)
        attended_visual = attention_weights * value  # (B, visual_dim)
        
        # Combine with LiDAR features
        combined = torch.cat([attended_visual, lidar_features], dim=-1)
        fused = self.output_proj(combined)
        
        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion: learnable gate to balance visual and LiDAR contributions
    """
    def __init__(self, visual_dim, lidar_dim, output_dim, verbose=True):
        super(GatedFusion, self).__init__()
        self.visual_dim = visual_dim
        self.lidar_dim = lidar_dim
        self.output_dim = output_dim
        
        # Project to same dimension
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.lidar_proj = nn.Linear(lidar_dim, output_dim)
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(visual_dim + lidar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # Gate values in [0, 1]
        )
        
        if verbose:
            print(f"[GatedFusion] visual_dim={visual_dim}, lidar_dim={lidar_dim}, output_dim={output_dim}")
        
    def forward(self, visual_features, lidar_features):
        """
        Args:
            visual_features: Visual features (B, visual_dim)
            lidar_features: LiDAR features (B, lidar_dim)
            
        Returns:
            fused_features: Fused features (B, output_dim)
        """
        # Project to output dimension
        visual_proj = self.visual_proj(visual_features)  # (B, output_dim)
        lidar_proj = self.lidar_proj(lidar_features)  # (B, output_dim)
        
        # Compute gate
        gate_input = torch.cat([visual_features, lidar_features], dim=-1)
        gate_values = self.gate(gate_input)  # (B, output_dim)
        
        # Fuse with learned gate
        fused = gate_values * visual_proj + (1 - gate_values) * lidar_proj
        
        return fused


def get_fusion_module(fusion_type='film', visual_dim=512, lidar_dim=256, output_dim=512, verbose=True):
    """
    Factory function to get fusion module
    
    Args:
        fusion_type: Type of fusion ('film', 'concat', 'attention', 'gated')
        visual_dim: Dimension of visual features
        lidar_dim: Dimension of LiDAR features
        output_dim: Dimension of fused features
        verbose: Whether to print initialization info
        
    Returns:
        Fusion module
    """
    fusion_type = fusion_type.lower()
    
    if fusion_type == 'film':
        return FiLMFusion(visual_dim, lidar_dim, output_dim, verbose=verbose)
    elif fusion_type == 'concat':
        return ConcatFusion(visual_dim, lidar_dim, output_dim, verbose=verbose)
    elif fusion_type == 'attention':
        return AttentionFusion(visual_dim, lidar_dim, output_dim, verbose=verbose)
    elif fusion_type == 'gated':
        return GatedFusion(visual_dim, lidar_dim, output_dim, verbose=verbose)
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}")


if __name__ == '__main__':
    # Test the fusion modules
    print("Testing fusion modules...")
    
    visual_features = torch.randn(4, 512)
    lidar_features = torch.randn(4, 256)
    
    # Test FiLM
    film = get_fusion_module('film', visual_dim=512, lidar_dim=256, output_dim=512)
    fused = film(visual_features, lidar_features)
    print(f"FiLM output shape: {fused.shape}")
    
    # Test Concat
    concat = get_fusion_module('concat', visual_dim=512, lidar_dim=256, output_dim=512)
    fused = concat(visual_features, lidar_features)
    print(f"Concat output shape: {fused.shape}")
    
    # Test Attention
    attention = get_fusion_module('attention', visual_dim=512, lidar_dim=256, output_dim=512)
    fused = attention(visual_features, lidar_features)
    print(f"Attention output shape: {fused.shape}")
    
    # Test Gated
    gated = get_fusion_module('gated', visual_dim=512, lidar_dim=256, output_dim=512)
    fused = gated(visual_features, lidar_features)
    print(f"Gated output shape: {fused.shape}")
    
    print("All tests passed!")
