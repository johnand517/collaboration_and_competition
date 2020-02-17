import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

n1 = 512  # Number of nodes in first layer of neural network
n2 = 256  # Number of nodes in 2nd layer of neural network
n3 = 128  # Number of nodes in the 3rd layer of neural network


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build Deep Q Network Model

        :param state_size: (int) Dimension of state space
        :param action_size: (int) Dimension of action space
        :param seed: (int) Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # First layer with dropout and Batch Normalization
        self.fc1 = nn.Linear(state_size, n1)
        self.bn = nn.BatchNorm1d(n1)
        self.fc1_drop = nn.Dropout(p=0.0)
        # Second layer with dropout
        self.fc2 = nn.Linear(n1, n2)
        self.fc2_drop = nn.Dropout(p=0.0)
        # Output layer
        self.fc3 = nn.Linear(n2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize model weights """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a network to map state to action values

        :param state: (array_like) the current state
        :return: (array_like) action outcomes from model
        """
        # Use Relu activation functions for hidden layers
        x = F.relu(self.fc1_drop(self.fc1(state)))
        x = self.bn(x)
        x = F.relu(self.fc2_drop(self.fc2(x)))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model"""

    def __init__(self, state_size, action_size, num_agents, seed):
        """Initialize parameters and build Deep Q Network Model

        :param state_size: (int) Dimension of state space
        :param action_size: (int) Dimension of action space
        :param seed: (int) Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # First layer with dropout
        self.fc1 = nn.Linear(state_size, n1)
        self.bn = nn.BatchNorm1d(n1)
        self.fc1_drop = nn.Dropout(p=0.0)
        # Second layer with dropout
        self.fc2 = nn.Linear(n1+action_size*num_agents, n2)  # Critic network accepts actions from *all* agents
        self.fc2_drop = nn.Dropout(p=0.0)
        # Third layer with dropout
        self.fc3 = nn.Linear(n2, n3)
        self.fc3_drop = nn.Dropout(p=0.0)
        # Output layer
        self.fc4 = nn.Linear(n3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize model weights """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a network to map state to action values

        :param state: (array_like) the current state
        :param action: (array_like) predicted action array
        :return: (array_like) action outcomes from model
        """
        # Use Relu activation functions for hidden layers
        xs = F.relu(self.fc1_drop(self.fc1(state)))
        xs = self.bn(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2_drop(self.fc2(x)))
        x = F.relu(self.fc3_drop(self.fc3(x)))
        return self.fc4(x)
