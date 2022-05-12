"""
This file contains a neural network module for us to
define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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
                 **kwargs):
        super(NetCritic, self).__init__()


        self.bn1 = nn.BatchNorm1d(in_dim)
        self.rb1 = ResBlock(in_dim, n_neurons)
        self.rb2 = ResBlock(n_neurons + in_dim, n_neurons)
        self.rb3 = ResBlock(n_neurons + in_dim, n_neurons)
        self.out = nn.Linear(n_neurons, out_dim)
        self.do = nn.Dropout(p=.1, inplace=False)

    def forward(self, obs):

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(device)

        X0 = obs
        # X0 = self.bn1(X)
        X = self.rb1(X0, True)
        X = self.rb2(torch.cat([X0, X], dim=-1), True)
        X = self.rb3(torch.cat([X0, X], dim=-1), True)
        output = self.out(X)
        return output


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
