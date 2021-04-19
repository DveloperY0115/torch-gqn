"""
Generation & Inference cores for GQN
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from conv_lstm import ConvLSTMCls

class GenerationCore(nn.Module):

    def __init__(self):
        super(GenerationCore, self).__init__()
        self.upsample_q = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)

    def forward(self, x):
        pass


class InferenceCore(nn.Module):

    def __init__(self):
        super(InferenceCore, self).__init__()
        self.upsample_q = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
    def forward(self, x):
        pass