"""
Unit-test for installation validation
"""

import os
import sys
import unittest

import torch
import tensorflow as tf

class envTest(unittest.TestCase):
    
    def test_runs(self):
        self.assertTrue(torch.__version__ == '1.8.1')    # should be 1.8.1
        self.assertTrue(tf.__version__ == '1.15.0')    # should be 1.15.0

if __name__ == '__main__':
    unittest.main()
