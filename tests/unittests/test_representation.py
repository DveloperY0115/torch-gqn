"""

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

from models.representation import PyramidCls, TowerCls, PoolCls

class TowerClsTest(unittest.TestCase):

    def test_runs(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TowerCls().to(device)

        summary(model, [(64, 64, 3), (1, 1, 7)])

class PoolClsTest(unittest.TestCase):

    def test_runs(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PoolCls().to(device)

        summary(model, [(64, 64, 3), (1, 1, 7)])

if __name__ == '__main__':
    unittest.main()