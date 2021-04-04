import os
import sys
import unittest

BASE_DIR = '../'    # path to project root directory
sys.path.append(BASE_DIR)

from utils.tfrecord_converter import TFRecordConverter

# file used for testing
TEST_FILENAME = '0001-of-2160.tfrecord'

class convert2numpyTest(unittest.TestCase):

    # TODO: Implement this testcase and check whether the output is correct!
    def test_runs(self):
        pass

if __name__ == '__main__':
    unittest.main()