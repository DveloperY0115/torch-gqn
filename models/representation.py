"""
Representation network of GQN
"""

import torch

class PyramidCls(nn.Module):

    def __init__(self):
        """
        Pyramid architecture for scene representation
        """

        super(PyramidCls, self).__init__()


    def forward(self, x, y):
        """
        Forward propagation

        Args:
        - x: A Tensor of shape (B, W, H, C). Batch of images
        - y: A Tensor of shape (B, 1, 1, 7). Batch of camera extrinsics

        Returns:
        """
        pass


class TowerCls(nn.Module):

    def __init__(self):
        """
        Tower architecture for scene representation
        """

        super(TowerCls, self):__init__()


    def forward(self, x, y):
        """
        Forward propagation

        Args:
        - x: A Tensor of shape (B, W, H, C). Batch of images
        - y: A Tensor of shape (B, 1, 1, 7). Batch of camera extrinsics

        Returns:
        """
        pass


class PoolCls(nn.Module):

    def __init__(self):
        """
        Pool architecture for scene representation
        """

        super(PoolCls, self).__init__()


    def forward(self, x, y):
        """
        Forward propagation

        Args:
        - x: A Tensor of shape (B, W, H, C). Batch of images
        - y: A Tensor of shape (B, 1, 1, 7). Batch of camera extrinsics

        Returns: 
        """
        pass