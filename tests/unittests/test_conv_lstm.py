"""
Unit test for generation architectures of GQN
"""

import os
import sys
import unittest
from pathlib import Path

import torch
from torchsummary import summary

# constants
BASE_DIR = os.path.join(Path(__file__).parent.absolute(), '../../')    # path to project root directory

sys.path.append(BASE_DIR)    # append project root to import paths

from models.conv_lstm import ConvLSTMCls

class ConvLSTMClsTest(unittest.TestCase):

    def test_runs(self):

        print('----------< Testing ConvLSTMCls >---------')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = ConvLSTMCls(256 * 2 + 128 + 7, 256, 128).to(device)

        # (dummy) input shapes for testing -> may behave differently in action
        query_shape = (7, 16, 16)    # should the cell remember query then how should we modify it?
        repr_shape = (256, 16, 16)
        latent_shape = (128, 16, 16)
        hidden_in_shape = (256, 16, 16)
        cell_in_shape = (256, 16, 16)
        skip_in_shape = (128, 64, 64)

        summary(model, [query_shape, repr_shape, latent_shape, hidden_in_shape, cell_in_shape, skip_in_shape])

if __name__ == '__main__':
   unittest.main()