"""
GQN
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from .representation import PyramidCls, TowerCls, PoolCls
from .core import GenerationCore, InferenceCore


class GQNCls(nn.Module):

    def __init__(self, repr_architecture='Tower', L=12, shared_core=False):
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
        self.L = L    # the number of generation cores

        # share core
        self.shared_core = shared_core

        if self.shared_core:
            self.gen_net = GenerationCore()
            self.inf_net = InferenceCore()
        else:
            self.gen_net = nn.ModuleList([GenerationCore() for _ in range(self.L)])
            self.inf_net = nn.ModuleList([InferenceCore() for _ in range(self.L)])
        
        # additional networks for Gaussian latent variable sampling
        self.eta_pi = nn.Conv2d(128, 6, kernel_size=5, stride=1, padding=2)
        self.eta_q = nn.Conv2d(128, 6, kernel_size=5, stride=1, padding=2)
        self.eta_g = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x, v, x_q, v_q, sigma_t):
        """
        Calculate ELBO, given the training data

        Args:
        - x: A Tensor of shape (B, S, W, H, C). Batch of image sequences
        - v: A Tensor of shape (B, S, 7, 1, 1). Batch of viewpoint sequences
        - x_q: A Tensor of shape (B, W, H, C). Batch of query images
        - v_q: A Tensor of shape (B, 7, 1, 1). Batch of query viewpoints

        Returns: ELBO (Evidence Lower BOund) calculated from the input data
        """

        B, S, *_ = x.size()

        device = x.device

        # Encode scenes
        if self.repr_architecture == 'Tower':
            r = torch.zeros((B, 256, 16, 16), device=device)
        else:
            r = torch.zeros((B, 256, 1, 1), device=device)
        for s in range(S):
            r += self.repr_net(x[:, s, :, :, :], v[:, s, :, :, :])

        # initialize generation core states
        cell_g = torch.zeros((B, 128, 16, 16), device=device)
        hidden_g = torch.zeros((B, 128, 16, 16), device=device)
        u = torch.zeros((B, 128, 64, 64), device=device)

        # initialize inference core states
        cell_e = torch.zeros((B, 128, 16, 16), device=device)
        hidden_e = torch.zeros((B, 128, 16, 16), device=device)

        # initialize ELBO
        elbo = 0

        # recurrent loop
        for l in range(self.L):
            # prior factor
            mu_pi, std_pi = torch.split(self.eta_pi(hidden_g), 3, dim=1)
            std_pi = torch.exp(0.5 * std_pi)    # standard deviation >= 0
            pi = Normal(mu_pi, std_pi)    # prior distribution

            # inference state update
            if self.shared_core:
                hidden_e, cell_e = self.inf_net(v_q, x_q, r, hidden_g, hidden_e, cell_e, u)
            else:
                hidden_e, cell_e = self.inf_net[l](v_q, x_q, r, hidden_g, hidden_e, cell_e, u)

            # posterior factor
            mu_q, std_q = torch.split(self.eta_q(hidden_e), 3, dim=1)
            std_q = torch.exp(0.5 * std_q)    # standard deviation >= 0
            q = Normal(mu_q, std_q)

            # sample posterior latent variable
            z = q.rsample()

            # update generator state
            if self.shared_core:
                hidden_g, cell_g, u = self.gen_net(v_q, r, z, hidden_g, cell_g, u)
            else:
                hidden_g, cell_g, u = self.gen_net[l](v_q, r, z, hidden_g, cell_g, u)

            # update KL contribution (regularization term) to the ELBO
            kl_div = kl_divergence(q, pi)

            # ELBO <- ELBO - KL(...)
            # elbo -= torch.sum(kl_div, dim=[1, 2, 3])
            elbo -= kl_div.sum(dim=[1, 2, 3])

        total_kl_div = elbo

        # calculate log likelihood contribution
        mu = self.eta_g(u)
        # likelihood = torch.sum(Normal(mu, sigma_t).log_prob(x_q), dim=[1, 2, 3])
        likelihood = Normal(mu, sigma_t).log_prob(x_q).sum(dim=[1, 2, 3])
        elbo += likelihood

        return elbo, total_kl_div, likelihood

    def generate(self, x, v, v_q):
        """
        Generate target image given the sequence of images from the scene,
        camera extrinsic, and the query viewpoint

        Args:
        - x: A Tensor of shape (B, S, W, H, C). Images from the single scene
        - v: A Tensor of shape (B, 7, 1, 1). Camera extrinsic corresponds to x
        - v_q: A Tensor of shape (B, 7, 1, 1). Query viewpoint
        - sigma_t: A scalar. Annealed pixel-wise variance

        Returns:
        - A target image would been seen at the query viewpoint v_q
        """

        B, S, *_ = x.size()

        device = x.device

        # Encode scenes
        if self.repr_architecture == 'Tower':
            r = torch.zeros((B, 256, 16, 16), device=device)
        else:
            r = torch.zeros((B, 256, 1, 1), device=device)
        for s in range(S):
            r += self.repr_net(x[:, s, :, :, :], v[:, s, :, :, :])

        # initialize generation core states
        cell_g = torch.zeros((B, 128, 16, 16), device=device)
        hidden_g = torch.zeros((B, 128, 16, 16), device=device)
        u = torch.zeros((B, 128, 64, 64), device=device)

        for l in range(self.L):
            # prior factor
            mean_pi, std_pi = torch.split(self.eta_pi(hidden_g), 3, dim=1)  # (B, 3, 16, 16) each
            std_pi = torch.exp(0.5 * std_pi)
            pi = Normal(mean_pi, std_pi)

            # sample prior latent variable
            z = pi.sample()

            # update generation core state
            if self.shared_core:
                hidden_g, cell_g, u = self.gen_net(v_q, r, z, hidden_g, cell_g, u)
            else:
                hidden_g, cell_g, u = self.gen_net[l](v_q, r, z, hidden_g, cell_g, u)

        # image sample
        mu = self.eta_g(u)

        # somehow, sampling doesn't work well!
        # img = Normal(mu, sigma_t).sample()

        # Return shape -> (B, 3, 64, 64)
        return torch.clamp(mu, 0, 1)
