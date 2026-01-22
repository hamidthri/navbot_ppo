"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
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
        
        # Trainable vision projection head (1280 -> 64)
        if self.use_vision:
            self.vision_proj = nn.Sequential(
                nn.Linear(vision_feat_dim, vision_proj_dim),
                nn.LayerNorm(vision_proj_dim),
                nn.LeakyReLU(negative_slope=0.2)
            )
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
