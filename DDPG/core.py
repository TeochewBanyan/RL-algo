# -*- coding: utf-8 -*-
"""
implement actor and critic network
"""

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

"""
If shape is scalar, use shape as the second dimension
else extend shape to the later dimensions
"""


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


"""
Pack the network.
The first n-2 layers are linear layers, with activation
The n-1 layer is output layers, with Identity as config act
'sizes' set the layer dimensions
nn.Identity: output the same as input

"""


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


"""
return the counts of all variables in the module
"""


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


"""
Policy net
Input: state
Output: action
params:
obs_dim, act_dim: state and action space
hidden_sizes: sizes of hidden layers
activation: activation function for hidden layers
act_limit: action space limits
"""


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.pi(obs)


"""
Q network

"""


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    """
  params:
  obs, act: tensor of state and action
  """

    def forward(self, obs, act):
        q = self.q(
            torch.cat([obs, act], dim=-1)
        )  # concat obs and act with the last dim
        return torch.squeeze(q, -1)  # q should be a scalar


"""
act_space.high?
"""


class MLPActorCritic(nn.Module):
    def __init__(
        self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU
    ):
        super().__init__()
        # init dimension info and action limit
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        act_limit = act_space.high[0]

        # init actor and q functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    # take action follow the policy network
    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

