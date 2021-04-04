"""
Unit-test for checking basic functionality of TFRecordConverter
"""

import os
import sys
import unittest
from pathlib import Path
import tensorflow as tf

# tensorflow settings
tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # suppress warnings

# constants
BASE_DIR = os.path.join(Path(__file__).parent.absolute(), '../../')    # path to project root directory
TEST_DIR = os.path.join(BASE_DIR, 'data/')
TEST_FILENAME = os.path.join(TEST_DIR, 'rooms_ring_camera/train/', '0001-of-2160.tfrecord')    # file used for testing
OUTPUT_DIR = os.path.join(Path(__file__).parent.absolute(), '/test_outputs')

sys.path.append(BASE_DIR)    # append project root to import paths

from utils.tfrecord_converter import TFRecordConverter
from utils.tfrecord_converter import DatasetInfo, _DATASETS

class convert2numpyTest(unittest.TestCase):

    def setUp(self):
        # set up environment before test starts
        if not os.path.exists('./test_outputs'):
            os.mkdir('./test_outputs')
        pass

    def test_runs(self):

        print('[!] TEST STARTED')
        info = _DATASETS['rooms_ring_camera']

        raw_data = tf.python_io.tf_record_iterator(TEST_FILENAME).__next__()
        converter = TFRecordConverter(info, TEST_DIR, 'train')

        frames, cameras = converter.convert_raw_to_numpy(raw_data, OUTPUT_DIR)

        # converting single .tfrecord file
        # images should be of size (10, 64, 64, 3)
        # cameras should be of size (10, 5)
        self.assertTrue(frames.shape == (10, 64, 64, 3))
        self.assertTrue(cameras.shape == (10, 5))

    def tearDown(self):
        # clean environment after testing
        import shutil
        if os.path.exists('./test_outputs'):
            shutil.rmtree('./test_outputs')

if __name__ == '__main__':
    unittest.main()
