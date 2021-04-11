"""
Unit test for generation architectures of GQN
"""

import os
import sys
import unittest
from pathlib import Path

from torchsummary import summary

# constants
BASE_DIR = os.path.join(Path(__file__).parent.absolute(), '../../')    # path to project root directory

sys.path.append(BASE_DIR)    # append project root to import paths

from models.generation import ConvLSTMCls, GenerationCls

class ConvLSTMClsTest(unittest.TestCase):

    def test_runs(self):
        pass

class GenerationClsTest(unittest.TestCase):

    def test_runs(self):
        pass


# if __name__ == '__main__':
#   unittest.main()