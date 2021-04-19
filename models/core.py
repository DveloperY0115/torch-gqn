"""
Generation & Inference cores for GQN
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from conv_lstm import ConvLSTMCls

class GenerationCore(nn.Module):

    def __init__(self):
        """
        Generation core of GQN
        """

        super(GenerationCore, self).__init__()

        self.upsample_q = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
        self.conv_lstm = ConvLSTMCls()

    def forward(self, q, r, z, hidden_in, cell_in, skip_in):
        """
        Forward propagation

        Similar to that of ConvLSTM, but includes upsampling & downsampling of input data

        Args:
        - q: A Tensor of shape (B, 7, 1, 1). Camera extrinsics of query view point
        - r: A Tensor of shape (B, C, W, H). Contains scene representation. Usually of shape (B, 256, 16, 16) or (B, 256, 1, 1)
        - z: A Tensor of shape (B, C, 16, 16). Upsampled latent vector from inference architecture
        - hidden_in: A Tensor of shape (B, C, W, H). Hidden variable from previous ConvLSTM cell. Usually of shape (B, 256, 16, 16)
        - cell_in: A Tensor of shape (B, C, W, H). Cell state from previous ConvLSTM cell. Usually of shape (B, 256, 16, 16)
        - skip_in: A Tensor of shape (B, C, W, H). Skip connection from previous ConvLSTM cell. Usually of shape (B, 128, 64, 64)

        Returns: A tuple of Tensors.
        - hidden: A Tensor of shape (B, C, W, H). Usually of shape (B, 256, 16, 16)
        - cell: A Tensor of shape (B, C, W, H). Usually of shape (B, 256, 16, 16)
        - skip: A Tensor of shape (B, C, W, H). Usually of shape (B, 128, 64, 64)
        """

        # upsample or downsample data if needed
        q = self.upsample_q(q)    # (B, 7, 1, 1) -> (B, 7, 16, 16)
        
        pass


class InferenceCore(nn.Module):

    def __init__(self):
        """
        Inference core of GQN
        """

        super(InferenceCore, self).__init__()

        self.upsample_q = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
    
    def forward(self, x):
        pass