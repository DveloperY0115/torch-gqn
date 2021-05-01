"""
Convolutional LSTM cell
"""

import torch
import torch.nn as nn

class ConvLSTMCls(nn.Module):

    def __init__(self, in_channels, out_channels, skip_out_channels):
        """
        Convolutional LSTM block for generation network

        Args:
        - in_channels: Int. Number of channels of the input of Conv2D
        - out_channels: Int. Number of channels of the result of Conv2D
        - skip_out_channels: Int. Number of channels of the result of transposed Conv2D in skip connection
        """

        super(ConvLSTMCls, self).__init__()

        # Conv layers for each gate of LSTM cell (size preserving)
        self.forget_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.input_conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.input_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

        # Conv layer for skip-connection
        self.skip_conv = nn.ConvTranspose2d(out_channels, skip_out_channels, kernel_size=4, stride=4)

    def forward(self, q, r, z, hidden_in, cell_in, skip_in):
        """
        Forward propagation

        Args:
        - q: A Tensor of shape (B, 7, 1, 1). Camera extrinsic of query view point
        - r: A Tensor of shape (B, C, W, H). Contains scene representation. Usually of shape (B, 256, 16, 16) or (B, 256, 1, 1)
        - z: A Tensor of shape (B, C, 16, 16). Up-sampled latent vector from inference architecture
        - hidden_in: A Tensor of shape (B, C, W, H). Hidden variable from previous ConvLSTM cell. Usually of shape (B, 256, 16, 16)
        - cell_in: A Tensor of shape (B, C, W, H). Cell state from previous ConvLSTM cell. Usually of shape (B, 256, 16, 16)
        - skip_in: A Tensor of shape (B, C, W, H). Skip connection from previous ConvLSTM cell. Usually of shape (B, 128, 64, 64)

        Returns: A tuple of Tensors.
        - hidden: A Tensor of shape (B, C, W, H). Usually of shape (B, 256, 16, 16)
        - cell: A Tensor of shape (B, C, W, H). Usually of shape (B, 256, 16, 16)
        - skip: A Tensor of shape (B, C, W, H). Usually of shape (B, 128, 64, 64)
        """
        
        # concatenate inputs along C dimension
        q = q.repeat(1, 1, 16, 16)
        x = torch.cat((hidden_in, q, r, z), dim=1)

        forget_gate = torch.sigmoid(self.forget_conv(x))
        input_gate = torch.sigmoid(self.input_conv_1(x)) * torch.tanh(self.input_conv_2(x))
        output_gate = torch.sigmoid(self.output_conv(x))

        cell = forget_gate * cell_in + input_gate
        hidden = output_gate * torch.tanh(cell)
        skip = self.skip_conv(hidden) + skip_in

        return hidden, cell, skip