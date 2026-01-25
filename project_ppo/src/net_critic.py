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

# Import fusion modules from net_actor
from net_actor import ViTFiLMTokenLearnerFusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=512):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
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

class NetCritic(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_neurons=512,
                 use_vision=False,
                 vision_feat_dim=1280,
                 vision_proj_dim=64,
                 **kwargs):
        super(NetCritic, self).__init__()

        self.use_vision = use_vision
        self.vision_feat_dim = vision_feat_dim
        self.vision_proj_dim = vision_proj_dim
        self.base_state_dim = in_dim
        
        # Trainable vision projection head (vision_feat_dim -> vision_proj_dim)
        if self.use_vision:
            self.vision_proj = ProjectionMLP(vision_feat_dim, vision_proj_dim, dropout=0.1)
            residual_input_dim = self.base_state_dim + vision_proj_dim
            print(f"[NetCritic] Vision mode: base_state={self.base_state_dim}, "
                  f"vision_proj={vision_proj_dim}, fused={residual_input_dim}", flush=True)
        else:
            self.vision_proj = None
            residual_input_dim = in_dim

        # Existing residual MLP head - use fused dimension
        self.bn1 = nn.BatchNorm1d(residual_input_dim)
        self.rb1 = ResBlock(residual_input_dim, residual_input_dim)
        self.rb2 = ResBlock(residual_input_dim + residual_input_dim, residual_input_dim + residual_input_dim)
        self.out = nn.Linear(residual_input_dim + residual_input_dim, out_dim)
        self.do = nn.Dropout(p=.1, inplace=False)

    def forward(self, obs, vision_feat=None):
        """
        Forward pass with optional vision features.
        
        Args:
            obs: Base state vector (B, base_state_dim) or (base_state_dim,)
            vision_feat: Vision features (B, 1280) or (1280,) from frozen backbone (optional)
        
        Returns:
            value: (B, out_dim)
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
        output = self.out(X)
        return output


class ViTFiLMTokenLearnerCritic(nn.Module):
    """
    Critic with ViT tokens + FiLM + TokenLearner fusion.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_neurons=512,
                 use_vision=False,
                 vision_feat_dim=384,
                 num_learned_tokens=8,
                 vision_emb_dim=128,
                 **kwargs):
        super().__init__()
        
        self.use_vision = use_vision
        self.base_state_dim = in_dim
        
        if not self.use_vision:
            raise ValueError("ViTFiLMTokenLearnerCritic requires use_vision=True")
        
        # Fusion
        self.fusion = ViTFiLMTokenLearnerFusion(
            base_state_dim=in_dim,
            token_dim=vision_feat_dim,
            num_learned_tokens=num_learned_tokens,
            vision_emb_dim=vision_emb_dim
        )
        
        fused_dim = in_dim + vision_emb_dim
        
        # Value head
        self.rb1 = ResBlock(fused_dim, fused_dim, n_neurons)
        self.rb2 = ResBlock(fused_dim + fused_dim, fused_dim + fused_dim, n_neurons)
        self.out = nn.Linear(fused_dim + fused_dim, out_dim)
        
        print(f"[ViTFiLMTokenLearnerCritic] base={in_dim}, vision_emb={vision_emb_dim}, fused={fused_dim}", flush=True)
    
    def forward(self, obs, vision_tokens=None):
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
        X0 = self.fusion(vision_tokens, obs)
        
        # Value head
        X = self.rb1(X0, True)
        X = self.rb2(torch.cat([X0, X], dim=-1), True)
        output = self.out(X)
        
        if needs_unsqueeze:
            output = output.squeeze(0)
        
        return output


class RecurrentViTFiLMTokenLearnerCritic(nn.Module):
    """
    Recurrent Critic with GRU.
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
        self.gru_hidden_dim = gru_hidden_dim
        
        if not self.use_vision:
            raise ValueError("RecurrentViTFiLMTokenLearnerCritic requires use_vision=True")
        
        # Fusion
        self.fusion = ViTFiLMTokenLearnerFusion(
            base_state_dim=in_dim,
            token_dim=vision_feat_dim,
            num_learned_tokens=num_learned_tokens,
            vision_emb_dim=vision_emb_dim
        )
        
        fused_dim = in_dim + vision_emb_dim
        
        # GRU
        self.gru = nn.GRU(fused_dim, gru_hidden_dim, batch_first=True)
        
        # Value head
        self.rb1 = ResBlock(gru_hidden_dim, gru_hidden_dim, n_neurons)
        self.rb2 = ResBlock(gru_hidden_dim + gru_hidden_dim, gru_hidden_dim + gru_hidden_dim, n_neurons)
        final_dim = gru_hidden_dim + gru_hidden_dim
        self.out = nn.Linear(final_dim, out_dim)
        
        self.hidden = None
        
        print(f"[RecurrentViTFiLMTokenLearnerCritic] gru_hidden={gru_hidden_dim}", flush=True)
    
    def forward(self, obs, vision_tokens=None, done=None):
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
        fused = self.fusion(vision_tokens, obs)
        
        # GRU
        if self.hidden is None or self.hidden.shape[1] != B:
            self.hidden = torch.zeros(1, B, self.gru_hidden_dim, device=device)
        
        if done is not None:
            if isinstance(done, np.ndarray):
                done = torch.tensor(done, dtype=torch.float, device=device)
            done = done.to(device).view(1, B, 1)
            self.hidden = self.hidden * (1 - done)
        
        fused_seq = fused.unsqueeze(1)
        gru_out, self.hidden = self.gru(fused_seq, self.hidden)
        gru_out = gru_out.squeeze(1)
        
        # Value head
        X = self.rb1(gru_out, True)
        X = self.rb2(torch.cat([gru_out, X], dim=-1), True)
        output = self.out(X)
        
        if needs_unsqueeze:
            output = output.squeeze(0)
        
        return output
    
    def reset_hidden(self):
        self.hidden = None


class NetCritic_old(nn.Module):

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
        self.layer3 = nn.Linear(64, out_dim)

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

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = F.tanh(self.layer3(activation2))
        return output
