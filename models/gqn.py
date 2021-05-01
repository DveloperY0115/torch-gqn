"""
GQN
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from .representation import PyramidCls, TowerCls, PoolCls
from .core import GenerationCore, InferenceCore


class GQNCls(nn.Module):

    def __init__(self, repr_architecture='Tower', levels=12, shared_core=False):
        """
        Entire GQN architecture formed by combining
        representation network and generation network

        Args:
        - repr_architecture: String. Can be 'Pyramid', 'Tower', or 'Pool'. Determine the architecture of representation
        to be used
        """

        super(GQNCls, self).__init__()

        # initialize representation network
        architectures = ['Pyramid', 'Tower', 'Pool']

        if repr_architecture not in architectures:
            raise ValueError('[!] Representation network can be \'Pyramid\', \'Tower\', or \'Pool\'.')

        self.repr_architecture = repr_architecture

        if self.repr_architecture == 'Pyramid':
            self.repr_net = PyramidCls()
        elif self.repr_architecture == 'Tower':
            self.repr_net = TowerCls()
        else:
            self.repr_net = PoolCls()

        # initialize generation network
        self.levels = levels    # the number of generation cores

        if shared_core:
            self.gen_net = GenerationCore()
            self.inf_net = InferenceCore()
        else:
            self.gen_net = nn.ModuleList([GenerationCore() for i in range(levels)])
            self.inf_net = nn.ModuleList([InferenceCore() for i in range(levels)])

    def forward(self, x, v, x_q, v_q):
        """
        Forward propagation. Calculate ELBO(evidence lower bound).

        Args:
        - x: A Tensor of shape (B, S, W, H, C). Batch of image sequences
        - v: A Tensor of shape (B, S, 7, 1, 1). Batch of viewpoint sequences
        - x_q: A Tensor of shape (B, W, H, C). Batch of query images
        - v_q: A Tensor of shape (B, 7, 1, 1). Batch of query viewpoints

        Returns:

        """

        batch_size, len_sequence, _, _, _ = x.size()

        # Encode scenes
        if self.repr_architecture == 'Tower':
            r = torch.zeros((batch_size, 256, 16, 16))
        else:
            r = torch.zeros((batch_size, 256, 1, 1))
        for b in range(batch_size):
            r[b] = self.repr_net(x[b, :], v[b, :])

        # r.shape => (B, 256, 16, 16)

        # initialize generation core states
        cell_g = torch.zeros((batch_size, 128, 16, 16))
        hidden_g = torch.zeros((batch_size, 128, 16, 16))
        skip = torch.zeros((batch_size, 128, 64, 64))

        # initialize inference core states
        cell_i = torch.zeros((batch_size, 128, 16, 16))
        hidden_i = torch.zeros((batch_size, 128, 16, 16))



