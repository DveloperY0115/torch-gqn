"""
Representation network of GQN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        super(TowerCls, self).__init__()

        # Conv layers
        self.conv_1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv_4 = nn.Conv2d(263, 128, kernel_size=3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        # Conv layers for skip connections
        self.conv_skip_1 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv_skip_2 = nn.Conv2d(263, 256, kernel_size=3, stride=1, padding=1)

        # Batch Norm layers
        self.bn_1 = nn.BatchNorm2d(256)
        self.bn_2 = nn.BatchNorm2d(128)
        self.bn_3 = nn.BatchNorm2d(256)
        self.bn_4 = nn.BatchNorm2d(128)
        self.bn_5 = nn.BatchNorm2d(256)
        self.bn_6 = nn.BatchNorm2d(256)

    def forward(self, x, y):
        """
        Forward propagation

        Args:
        - x: A Tensor of shape (B, W, H, C). Batch of images
        - y: A Tensor of shape (B, 1, 1, 7). Batch of camera extrinsics

        Returns:
        """

        if not torch.is_tensor(x):
            x = torch.Tensor(x)

        if not torch.is_tensor(y):
            y = torch.Tensor(y)

        x = x.transpose(1, 3)    # x.shape = (B, C, H, W)

        x = F.relu(self.bn_1(self.conv_1(x)))    # x.shape = (B, 256, 32, 32)

        # Residual connection
        skip = self.conv_skip_1(x)    # skip.shape = (B, 256, 16, 16)

        x = F.relu(self.bn_2(self.conv_2(x)))    # x.shape = (B, 128, 32, 32)
        x = F.relu(self.bn_3(self.conv_3(x) + skip))    # x.shape = (B, 256, 16, 16)

        # Concatenate view point information
        y = y.transpose(1, 3)    # y.shape = (B, 7, 1, 1)
        y = y.repeat(1, 1, 16, 16)    # y.shape = (B, 7, 16, 16)
        
        x = torch.cat((x, y), dim=1)    # x.shape = (B, 263, 16, 16)

        # Residual connection
        skip = self.conv_skip_2(x)

        x = F.relu(self.bn_4(self.conv_4(x)))

        x = F.relu(self.bn_5(self.conv_5(x) + skip))

        x = F.relu(self.bn_6(self.conv_6(x)))

        # x.shape = (B, 256, 16, 16)
        return x


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