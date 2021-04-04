import os
import sys
import unittest

from ..utils.tfrecord_converter import TFRecordConverter

BASE_DIR = '../'    # path to project root directory
sys.path.append(BASE_DIR)

class convert2numpyTest(unittest.TestCase):

    def test_runs(self):
        pass

if __name__ == '__main__':
    unittest.main()