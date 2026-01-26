"""
Fusion Modules for Vision-LiDAR Integration
============================================

This module contains all fusion components for combining vision features 
(tokens or pooled) with base state (LiDAR + past actions + goal).

Components:
- TokenLearner: Reduce N spatial tokens to K learned tokens
- FiLMLayer: Feature-wise Linear Modulation conditioned on base state
- ViTFiLMTokenLearnerFusion: Complete token fusion pipeline (ViT/DINOv2)
- PooledFiLMFusion: FiLM-based fusion for pooled CNN features (ResNet/MobileNet)

All fusion modules output a fused vector: (B, base_state_dim + vision_emb_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLearner(nn.Module):
    """
    TokenLearner: Reduce N tokens to K learned tokens via weighted attention.
    Paper: https://arxiv.org/abs/2106.11297
    
    Simple, cheap implementation:
    - MLP over token channels produces K attention maps
    - Softmax over N (spatial) dimension
    - Weighted sum of original tokens
    """
    def __init__(self, token_dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        # MLP: C -> K attention logits per token
        self.attention_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, num_tokens),
        )
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, C) input tokens
        Returns:
            learned_tokens: (B, K, C) reduced tokens
        """
        B, N, C = tokens.shape
        
        # Compute attention scores: (B, N, C) -> (B, N, K)
        attn_logits = self.attention_mlp(tokens)  # (B, N, K)
        
        # Softmax over N (spatial dimension) for each of K tokens
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, N, K)
        
        # Weighted sum: (B, N, K)^T @ (B, N, C) -> (B, K, C)
        learned_tokens = torch.einsum('bnk,bnc->bkc', attn_weights, tokens)
        
        return learned_tokens


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioned on base state.
    
    base_state -> MLP -> (γ, β)
    tokens_out = γ * tokens + β
    
    Uses stable initialization: γ starts near 1, β near 0
    """
    def __init__(self, base_state_dim: int, token_dim: int):
        super().__init__()
        hidden_dim = max(64, base_state_dim * 2)
        
        # MLP: base_state -> (gamma, beta)
        self.film_mlp = nn.Sequential(
            nn.LayerNorm(base_state_dim),
            nn.Linear(base_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim * 2),  # γ and β
        )
        
        # Initialize to near-identity: γ ≈ 1, β ≈ 0
        # Last layer outputs Δγ and β, then γ = 1 + Δγ
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)
    
    def forward(self, tokens: torch.Tensor, base_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, K, C) tokens to modulate
            base_state: (B, base_state_dim) conditioning vector
        Returns:
            modulated_tokens: (B, K, C)
        """
        B, K, C = tokens.shape
        
        # Generate γ and β
        film_params = self.film_mlp(base_state)  # (B, 2*C)
        delta_gamma, beta = torch.split(film_params, C, dim=-1)  # (B, C), (B, C)
        
        # γ = 1 + Δγ for stability
        gamma = 1.0 + delta_gamma
        
        # Broadcast and apply: γ * tokens + β
        # gamma, beta: (B, C) -> (B, 1, C) for broadcasting
        gamma = gamma.unsqueeze(1)  # (B, 1, C)
        beta = beta.unsqueeze(1)    # (B, 1, C)
        
        modulated_tokens = gamma * tokens + beta
        
        return modulated_tokens


class ViTFiLMTokenLearnerFusion(nn.Module):
    """
    Complete fusion module for ViT/DINOv2 tokens: TokenLearner + FiLM + Readout
    
    Pipeline:
    1. Extract tokens from vision backbone (B, N, C)
    2. TokenLearner: (B, N, C) -> (B, K, C)
    3. FiLM: condition tokens on base_state
    4. Readout: flatten + MLP -> (B, vision_emb_dim)
    5. Fuse: concat(base_state_norm, vision_emb) -> (B, base_dim + vision_emb_dim)
    """
    def __init__(self, 
                 base_state_dim: int,
                 token_dim: int,  # C (e.g., 384 for dinov2_vits14)
                 num_learned_tokens: int = 8,  # K
                 vision_emb_dim: int = 128):
        super().__init__()
        self.base_state_dim = base_state_dim
        self.token_dim = token_dim
        self.num_learned_tokens = num_learned_tokens
        self.vision_emb_dim = vision_emb_dim
        
        # TokenLearner
        self.token_learner = TokenLearner(token_dim, num_learned_tokens)
        
        # FiLM
        self.film = FiLMLayer(base_state_dim, token_dim)
        
        # Readout MLP: (K * C) -> vision_emb_dim
        readout_input_dim = num_learned_tokens * token_dim
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_input_dim),
            nn.Linear(readout_input_dim, vision_emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(vision_emb_dim * 2, vision_emb_dim),
            nn.LayerNorm(vision_emb_dim),
        )
        
        # Base state normalization
        self.base_norm = nn.LayerNorm(base_state_dim)
    
    def forward(self, tokens: torch.Tensor, base_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, C) from vision backbone
            base_state: (B, base_state_dim)
        Returns:
            fused: (B, base_state_dim + vision_emb_dim)
        """
        # 1. TokenLearner: (B, N, C) -> (B, K, C)
        learned_tokens = self.token_learner(tokens)
        
        # 2. FiLM: condition on base_state
        modulated_tokens = self.film(learned_tokens, base_state)
        
        # 3. Readout: flatten + MLP
        B, K, C = modulated_tokens.shape
        flat_tokens = modulated_tokens.reshape(B, K * C)
        vision_emb = self.readout(flat_tokens)  # (B, vision_emb_dim)
        
        # 4. Fuse with normalized base state
        base_norm = self.base_norm(base_state)
        fused = torch.cat([base_norm, vision_emb], dim=-1)
        
        return fused


class PooledFiLMFusion(nn.Module):
    """
    FiLM-based fusion for pooled CNN features (ResNet, MobileNet, etc.)
    
    Replaces raw concatenation with learned FiLM modulation:
    - Projects pooled vision features to embedding space
    - Generates FiLM parameters (γ, β) from normalized base state
    - Modulates vision embedding with base-conditioned FiLM
    - Returns fused vector: concat(base_norm, modulated_vision_emb)
    
    This ensures vision and base state interact via learned conditioning
    rather than simple concatenation.
    """
    def __init__(self,
                 base_state_dim: int,
                 vision_feat_dim: int,  # Pooled feature dim (e.g., 512 for ResNet18)
                 vision_emb_dim: int = 64):  # Target embedding dimension
        super().__init__()
        self.base_state_dim = base_state_dim
        self.vision_feat_dim = vision_feat_dim
        self.vision_emb_dim = vision_emb_dim
        
        # Base state normalization
        self.base_norm = nn.LayerNorm(base_state_dim)
        
        # Vision projection: pooled features -> embedding
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, vision_emb_dim),
            nn.LayerNorm(vision_emb_dim),
        )
        
        # FiLM generator: base_state -> (γ, β) for vision embedding
        hidden_dim = max(64, base_state_dim * 2)
        self.film_mlp = nn.Sequential(
            nn.Linear(base_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vision_emb_dim * 2),  # γ and β
        )
        
        # Stable initialization: γ ≈ 1, β ≈ 0
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)
    
    def forward(self, vision_feat: torch.Tensor, base_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_feat: (B, vision_feat_dim) pooled features from CNN
            base_state: (B, base_state_dim) LiDAR + past actions + goal
        Returns:
            fused: (B, base_state_dim + vision_emb_dim) fused representation
        """
        B = base_state.shape[0]
        
        # Normalize base state
        base_norm = self.base_norm(base_state)  # (B, base_state_dim)
        
        # Project vision features
        vision_emb = self.vision_proj(vision_feat)  # (B, vision_emb_dim)
        
        # Generate FiLM parameters from base state
        film_params = self.film_mlp(base_norm)  # (B, 2 * vision_emb_dim)
        delta_gamma, beta = torch.split(film_params, self.vision_emb_dim, dim=-1)
        
        # Apply FiLM: γ = 1 + Δγ for stability
        gamma = 1.0 + delta_gamma  # (B, vision_emb_dim)
        modulated_vision = gamma * vision_emb + beta  # (B, vision_emb_dim)
        
        # Fuse: concatenate normalized base with modulated vision
        fused = torch.cat([base_norm, modulated_vision], dim=-1)
        
        return fused


def get_fused_dim(base_state_dim: int, vision_emb_dim: int) -> int:
    """
    Helper to compute fused dimension output by fusion modules.
    
    Args:
        base_state_dim: Dimension of base state (LiDAR + actions + goal)
        vision_emb_dim: Dimension of vision embedding after fusion
    
    Returns:
        fused_dim: base_state_dim + vision_emb_dim
    """
    return base_state_dim + vision_emb_dim
