"""
Representation network of GQN
"""

import torch
import torch.nn as nn
import torch.functional as F

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

        # Conv layers
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(2,2), stride=(2,2))
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2,2), stride=(2,2))
        self.conv_4 = nn.Conv2d(in_channels=263, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1))

        # Batch Norm layers
        self.bn_1 = nn.BatchNorm2d(256)
        self.bn_2 = nn.BatchNorm2d(128)
        self.bn_3 = nn.BatchNorm2d(256)
        self.bn_4 = nn.BatchNorm2d(128)
        self.bn_5 = nn.BatchNorm2d(256)
        # self.bn_6 = nn.BatchNorm2d(256) -> should it be used..?

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