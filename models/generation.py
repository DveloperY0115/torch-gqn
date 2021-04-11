"""
Generation network of GQN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GenerationCls(nn.Module):

    def __init__(self):
        """
        Generation network consists of multiple ConvLSTM units
        """

        super(GenerationCls, self).__init__()

    def forward(self, x):
        pass


class ConvLSTMCls(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Convolutional LSTM block for generation network

        Args:
        - in_channels: Int. Number of channels of the input of Conv2D
        - out_channels: Int. Number of channels of the result of Conv2D
        """

        super(ConvLSTMCls, self).__init__()
        
        # Conv layers for each gate of LSTM cell (size preserving)
        self.forget_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.input_conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.input_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4, padding=)
    
    def forward(self, q, r, z, hidden_in, cell_in, skip_in):
        """
        Forward propagation

        Args:
        - q:
        - r:
        - z:
        - hidden_in:
        - cell_in:
        - skip_in:

        Returns: A tuple of Tensors. The tuple includes hidden state, output, and skip connection of the cell
        """
        
        # concatenate inputs along C dimension
        x = torch.cat([hidden_in, q, r, z], dim=1)

        forget_gate = F.sigmoid(self.forget_conv(x))
        input_gate = F.sigmoid(self.input_conv_1(x)) * F.tanh(self.input_conv_2(x))
        output_gate = F.sigmoid(self.output_conv(x))

        cell = forget_gate * cell_in + input_gate
        hidden = output_gate * F.tanh(cell)
        skip = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=4) + skip_in 

        return (hidden, cell, skip)