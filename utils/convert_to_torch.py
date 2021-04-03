"""
Script for converting TFRecord format to Pytorch compatible formats
"""

import os
import sys
import collections
import tensorflow as tf

BASE_DIR = '../'    # path to project root directory
sys.path.append(BASE_DIR)

# define structure for dataset metadata
DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)

# dictionary of GQN-data metadata
_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)

# define named tuple for data processing
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])

def _convert_frame_data(jpeg_data):
    """
    Convert JPEG-encoded image to a uint8 tensor

    Args:
    - jpeg_data: A Tensor of type string. 0-D. The JPEG-encoded image

    Returns:
    - A Tensor of type float32
    """
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)

def _get_dataset_files(dataset_info, mode, root):
    """
    Generates lists of files for a given dataset information

    Args:
    - dataset_info: Named tuple. An object containing metadata of GQN datasets
    - mode: String. Can be 'train' or 'test'
    - root: String. Root directory of datasets

    Returns:
    - List of files in the dataset
    """
    basepath = dataset_info.basepath
    base = os.path.join(root, basepath, mode)

    if mode == 'train':
        num_files = dataset_info.train_size
    elif mode == 'test':
        num_files = dataset_info.test_size

    # usually of form '{}-of-{}.tfrecord'
    files = sorted(os.listdir(base))

    return [os.path.join(base, file) for file in files]

def make_context(frames, cameras):
    """
    Generate Context named tuple using camera, frame information

    Args:
    - cameras:
    - frames:

    Returns:
    - A Context named tuple encapsulating given information
    """
    return Context(cameras=cameras, frames=frames)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('[!] Please specify a dataset to convert')
        print('[!] python convert_to_torch.py dataset_name train/test/all')
        print('[!] e.g. python convert_to_torch.py jaco train')
        exit()

    DATASET = sys.argv[1]
    dataset_info = _DATASETS[DATASET]

    converted_dataset_path = f'{DATASET}-torch'

    # switches for converter
    convert_train = False
    convert_test = False

    if sys.argv[2] == 'train':
        convert_train = True
        converted_dataset_train = f'{converted_dataset_path}/train'
    
    elif sys.argv[2] == 'test':
        convert_test = True
        converted_dataset_test = f'{converted_dataset_path}/test'
    
    elif sys.argv[2] == 'all':
        convert_train = True
        convert_test = True
        converted_dataset_train = f'{converted_dataset_path}/train'
        converted_dataset_test = f'{converted_dataset_path}/test'

    else:
        print('[!] Please provide valid argument')
        exit()

    # make directories for store data files
    os.mkdir(converted_dataset_path)

    if convert_train:
        os.mkdir(converted_dataset_train)

    if convert_test:
        os.mkdir(converted_dataset_test)

    # convert train dataset
    if convert_train:
        pass

    # convert test datset
    if convert_test:
        pass