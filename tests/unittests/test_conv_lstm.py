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

        model = ConvLSTMCls(7+256+3, 128).to(device)

        # (dummy) input shapes for testing -> may behave differently in action
        input_shape = (7+256+3, 16, 16)
        hidden_in_shape = (128, 16, 16)
        cell_in_shape = (128, 16, 16)

        summary(model, [input_shape, hidden_in_shape, cell_in_shape])


if __name__ == '__main__':
   unittest.main()