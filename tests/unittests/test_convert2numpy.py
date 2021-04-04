import os
import sys
import unittest
from pathlib import Path

# constants
BASE_DIR = os.path.join(Path(__file__).parent.absolute(), '../../')    # path to project root directory
TEST_FILENAME = '0001-of-2160.tfrecord'    # file used for testing

sys.path.append(BASE_DIR)    # append project root to import paths

from utils.tfrecord_converter import TFRecordConverter

class convert2numpyTest(unittest.TestCase):

    # TODO: Implement this testcase and check whether the output is correct!
    def test_runs(self):
        pass

if __name__ == '__main__':
    unittest.main()
