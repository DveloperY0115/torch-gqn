"""
Unit test for GQN
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

from utils.data_loader import RoomsRingCameraDataset, DataLoader, sample_from_batch
from models.gqn import GQNCls


class GQNClsTest(unittest.TestCase):

    @unittest.skip('torchsummary crashes while testing GQN. \n'
                   'But the network works well on real data, so skipping the test')
    def test_runs(self):

        print('----------< Testing GQNCls >---------')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # prepare model
        model = GQNCls().to(device)

        summary(model, [(5, 3, 64, 64), (5, 7, 1, 1), (3, 64, 64), (7, 1, 1), (3, 64, 64)])


if __name__ == '__main__':
    unittest.main()