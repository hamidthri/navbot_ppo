"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from vision_backbones import ProjectionMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# TokenLearner + FiLM Fusion Modules
# ============================================================================

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
    Complete fusion module: TokenLearner + FiLM + Readout
    
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



class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=512):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        nn.init.uniform_(self.fc1.weight, -1 / math.sqrt(Fin), 1 / math.sqrt(Fin))
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        nn.init.uniform_(self.fc2.weight, -1 / math.sqrt(n_neurons), 1 / math.sqrt(n_neurons))
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        # Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        # Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class NetActor(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_neurons=512,
                 use_vision=False,
                 vision_feat_dim=1280,
                 vision_proj_dim=64,
                 **kwargs):
        super(NetActor, self).__init__()

        self.use_vision = use_vision
        self.vision_feat_dim = vision_feat_dim
        self.vision_proj_dim = vision_proj_dim
        self.base_state_dim = in_dim  # Base state (LiDAR + pose + past actions)
        
        # Trainable vision projection head (vision_feat_dim -> vision_proj_dim)
        if self.use_vision:
            self.vision_proj = ProjectionMLP(vision_feat_dim, vision_proj_dim, dropout=0.1)
            # Fused input: base_state + projected_vision
            residual_input_dim = self.base_state_dim + vision_proj_dim
            print(f"[NetActor] Vision mode: base_state={self.base_state_dim}, "
                  f"vision_proj={vision_proj_dim}, fused={residual_input_dim}", flush=True)
        else:
            self.vision_proj = None
            residual_input_dim = in_dim

        # Existing residual MLP head - use fused dimension
        self.bn1 = nn.BatchNorm1d(residual_input_dim)
        self.rb1 = ResBlock(residual_input_dim, residual_input_dim)
        self.rb2 = ResBlock(residual_input_dim + residual_input_dim, residual_input_dim + residual_input_dim)
        
        self.out1 = nn.Linear(residual_input_dim + residual_input_dim, out_dim - 1)
        nn.init.uniform_(self.out1.weight, -1 / math.sqrt(residual_input_dim), 1 / math.sqrt(residual_input_dim))
        self.out2 = nn.Linear(residual_input_dim + residual_input_dim, out_dim - 1)
        nn.init.uniform_(self.out2.weight, -1 / math.sqrt(residual_input_dim + residual_input_dim), 1 / math.sqrt(residual_input_dim + residual_input_dim))
        self.do = nn.Dropout(p=.1, inplace=False)

    def forward(self, obs, vision_feat=None):
        """
        Forward pass with optional vision features.
        
        Args:
            obs: Base state vector (B, base_state_dim) or (base_state_dim,)
            vision_feat: Vision features (B, 1280) or (1280,) from frozen backbone (optional)
        
        Returns:
            actions: (B, out_dim)
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(device)
        
        # Ensure obs has batch dimension
        needs_unsqueeze = (obs.dim() == 1)
        if needs_unsqueeze:
            obs = obs.unsqueeze(0)
        
        # Fuse vision features if provided
        if self.use_vision:
            if vision_feat is None:
                # Fallback: use zeros if no vision features provided
                vision_feat = torch.zeros(obs.shape[0], self.vision_feat_dim, dtype=torch.float, device=device)
            else:
                if isinstance(vision_feat, np.ndarray):
                    vision_feat = torch.tensor(vision_feat, dtype=torch.float)
                vision_feat = vision_feat.to(device)
                
                # Ensure vision_feat has batch dimension
                if vision_feat.dim() == 1:
                    vision_feat = vision_feat.unsqueeze(0)
            
            # Project vision features (trainable)
            vision_proj = self.vision_proj(vision_feat)
            
            # Concatenate base state + projected vision
            X0 = torch.cat([obs, vision_proj], dim=-1)
        else:
            X0 = obs

        # Existing residual head (unchanged)
        # X0 = self.bn1(X0)
        X = self.rb1(X0, True)
        X = self.rb2(torch.cat([X0, X], dim=-1), True)

        output1 = F.sigmoid(self.out1(X))
        output2 = F.tanh(self.out2(X))
        output = torch.cat((output1, output2), -1)
        return output


class ViTFiLMTokenLearnerActor(nn.Module):
    """
    Actor with ViT tokens + FiLM + TokenLearner fusion.
    
    Expects vision backbone to output tokens (B, N, C), not pooled features.
    Uses TokenLearner to reduce to K tokens, applies FiLM conditioning,
    then fuses with base state for policy.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_neurons=512,
                 use_vision=False,
                 vision_feat_dim=384,  # Token channel dim (C) for dinov2_vits14
                 num_learned_tokens=8,  # K
                 vision_emb_dim=128,
                 **kwargs):
        super().__init__()
        
        self.use_vision = use_vision
        self.base_state_dim = in_dim
        self.out_dim = out_dim
        self.vision_feat_dim = vision_feat_dim
        self.num_learned_tokens = num_learned_tokens
        self.vision_emb_dim = vision_emb_dim
        
        if not self.use_vision:
            raise ValueError("ViTFiLMTokenLearnerActor requires use_vision=True")
        
        # FiLM + TokenLearner fusion
        self.fusion = ViTFiLMTokenLearnerFusion(
            base_state_dim=in_dim,
            token_dim=vision_feat_dim,
            num_learned_tokens=num_learned_tokens,
            vision_emb_dim=vision_emb_dim
        )
        
        # Fused dimension: base_state_dim + vision_emb_dim
        fused_dim = in_dim + vision_emb_dim
        
        # Residual policy head
        self.rb1 = ResBlock(fused_dim, fused_dim, n_neurons)
        self.rb2 = ResBlock(fused_dim + fused_dim, fused_dim + fused_dim, n_neurons)
        
        self.out1 = nn.Linear(fused_dim + fused_dim, out_dim - 1)
        nn.init.uniform_(self.out1.weight, -1/math.sqrt(fused_dim), 1/math.sqrt(fused_dim))
        self.out2 = nn.Linear(fused_dim + fused_dim, out_dim - 1)
        nn.init.uniform_(self.out2.weight, -1/math.sqrt(fused_dim), 1/math.sqrt(fused_dim))
        
        print(f"[ViTFiLMTokenLearnerActor] base={in_dim}, tokens={num_learned_tokens}, "
              f"token_dim={vision_feat_dim}, vision_emb={vision_emb_dim}, fused={fused_dim}", flush=True)
    
    def forward(self, obs, vision_tokens=None):
        """
        Args:
            obs: (B, base_state_dim) or (base_state_dim,)
            vision_tokens: (B, N, C) tokens from backbone, or None
        Returns:
            actions: (B, out_dim)
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(device)
        
        needs_unsqueeze = (obs.dim() == 1)
        if needs_unsqueeze:
            obs = obs.unsqueeze(0)
        
        if vision_tokens is None:
            # Fallback: zero tokens
            B = obs.shape[0]
            # Assume 16x16 patches for 224x224 image at patch_size=14 -> ~16x16 = 256 tokens
            N = 256
            vision_tokens = torch.zeros(B, N, self.vision_feat_dim, device=device)
        else:
            if isinstance(vision_tokens, np.ndarray):
                vision_tokens = torch.tensor(vision_tokens, dtype=torch.float)
            vision_tokens = vision_tokens.to(device)
            
            if vision_tokens.dim() == 2:
                # (N, C) -> (1, N, C)
                vision_tokens = vision_tokens.unsqueeze(0)
        
        # Fusion: TokenLearner + FiLM + readout
        X0 = self.fusion(vision_tokens, obs)  # (B, fused_dim)
        
        # Residual policy head
        X = self.rb1(X0, True)
        X = self.rb2(torch.cat([X0, X], dim=-1), True)
        
        output1 = F.sigmoid(self.out1(X))
        output2 = F.tanh(self.out2(X))
        output = torch.cat((output1, output2), -1)
        
        if needs_unsqueeze:
            output = output.squeeze(0)
        
        return output


class RecurrentViTFiLMTokenLearnerActor(nn.Module):
    """
    Recurrent version with GRU memory after FiLM fusion.
    
    Pipeline:
    1. FiLM+TokenLearner fusion -> fused_dim
    2. GRU: (fused_dim) -> (gru_hidden_dim)
    3. Policy head from GRU output
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_neurons=512,
                 use_vision=False,
                 vision_feat_dim=384,
                 num_learned_tokens=8,
                 vision_emb_dim=128,
                 gru_hidden_dim=128,
                 **kwargs):
        super().__init__()
        
        self.use_vision = use_vision
        self.base_state_dim = in_dim
        self.out_dim = out_dim
        self.gru_hidden_dim = gru_hidden_dim
        
        if not self.use_vision:
            raise ValueError("RecurrentViTFiLMTokenLearnerActor requires use_vision=True")
        
        # Fusion
        self.fusion = ViTFiLMTokenLearnerFusion(
            base_state_dim=in_dim,
            token_dim=vision_feat_dim,
            num_learned_tokens=num_learned_tokens,
            vision_emb_dim=vision_emb_dim
        )
        
        fused_dim = in_dim + vision_emb_dim
        
        # GRU for temporal memory
        self.gru = nn.GRU(fused_dim, gru_hidden_dim, batch_first=True)
        
        # Policy head from GRU output
        self.rb1 = ResBlock(gru_hidden_dim, gru_hidden_dim, n_neurons)
        self.rb2 = ResBlock(gru_hidden_dim + gru_hidden_dim, gru_hidden_dim + gru_hidden_dim, n_neurons)
        
        final_dim = gru_hidden_dim + gru_hidden_dim
        self.out1 = nn.Linear(final_dim, out_dim - 1)
        self.out2 = nn.Linear(final_dim, out_dim - 1)
        
        # Hidden state (will be reset on episode boundaries)
        self.hidden = None
        
        print(f"[RecurrentViTFiLMTokenLearnerActor] base={in_dim}, gru_hidden={gru_hidden_dim}, "
              f"fused={fused_dim}", flush=True)
    
    def forward(self, obs, vision_tokens=None, done=None):
        """
        Args:
            obs: (B, base_state_dim)
            vision_tokens: (B, N, C)
            done: (B,) done flags to reset hidden state
        Returns:
            actions: (B, out_dim)
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(device)
        
        needs_unsqueeze = (obs.dim() == 1)
        if needs_unsqueeze:
            obs = obs.unsqueeze(0)
        
        B = obs.shape[0]
        
        if vision_tokens is None:
            N = 256
            vision_tokens = torch.zeros(B, N, self.vision_feat_dim, device=device)
        else:
            if isinstance(vision_tokens, np.ndarray):
                vision_tokens = torch.tensor(vision_tokens, dtype=torch.float)
            vision_tokens = vision_tokens.to(device)
            if vision_tokens.dim() == 2:
                vision_tokens = vision_tokens.unsqueeze(0)
        
        # Fusion
        fused = self.fusion(vision_tokens, obs)  # (B, fused_dim)
        
        # Initialize or reset hidden state
        if self.hidden is None or self.hidden.shape[1] != B:
            self.hidden = torch.zeros(1, B, self.gru_hidden_dim, device=device)
        
        if done is not None:
            # Reset hidden for done episodes
            if isinstance(done, np.ndarray):
                done = torch.tensor(done, dtype=torch.float, device=device)
            done = done.to(device).view(1, B, 1)  # (1, B, 1)
            self.hidden = self.hidden * (1 - done)
        
        # GRU: input (B, 1, fused_dim), hidden (1, B, gru_hidden)
        fused_seq = fused.unsqueeze(1)  # (B, 1, fused_dim)
        gru_out, self.hidden = self.gru(fused_seq, self.hidden)
        gru_out = gru_out.squeeze(1)  # (B, gru_hidden)
        
        # Policy head
        X = self.rb1(gru_out, True)
        X = self.rb2(torch.cat([gru_out, X], dim=-1), True)
        
        output1 = F.sigmoid(self.out1(X))
        output2 = F.tanh(self.out2(X))
        output = torch.cat((output1, output2), -1)
        
        if needs_unsqueeze:
            output = output.squeeze(0)
        
        return output
    
    def reset_hidden(self):
        """Reset GRU hidden state (call at episode start)"""
        self.hidden = None


class NetActor_old(nn.Module):

    """
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""

    def __init__(self, in_dim, out_dim):
        """
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
        super(self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim - 1)
        self.layer4 = nn.Linear(64, out_dim - 1)

    def forward(self, obs):
        """
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(device)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output1 = F.sigmoid(self.layer3(activation2))
        output2 = F.tanh(self.layer4(activation2))
        output = torch.cat((output1, output2), -1)
        return output
