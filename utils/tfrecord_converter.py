"""
Script for converting TFRecord format to Pytorch compatible formats
"""

import os
import sys
import collections
import pickle
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# suppress deprecation warning
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

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

# constants
_NUM_POSE_PARAMS = 5

def convert_raw_to_numpy(dataset_info, raw_data, path=None):
    """
    Convert raw data (image, camera in/extrinsics) to numpy and save it

    Args:
    - dataset_info: Named tuple. An object containing metadata of GQN datasets
    - raw_data: A scalar string Tensor. A single serialized example
    - path: String. Path where the converted data is stored
    """
    features= {
            'frames': tf.FixedLenFeature(
                shape=dataset_info.sequence_size, dtype=tf.string),
            'cameras': tf.FixedLenFeature(
                shape=[dataset_info.sequence_size * 5],
                dtype=tf.float32)
        }

    example = tf.parse_single_example(raw_data, features)
    frames = _process_frames(dataset_info, example)
    cameras = _process_cameras(dataset_info, example, True)

    context = [frames.numpy().squeeze(), cameras.numpy().squeeze()]

    if path is not None:
        with open(path, 'wb') as f:
            pickle.dump(context, f)

    # return frames.numpy().squeeze(), cameras.numpy().squeeze()

def _convert_frame_data(jpeg_data):
    """
    Convert JPEG-encoded image to a uint8 tensor

    Args:
    - jpeg_data: A Tensor of type string. 0-D. The JPEG-encoded image

    Returns: A Tensor of type float32
    """
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)

def get_dataset_files(dataset_info, mode, root):
    """
    Generates lists of files for a given dataset information

    Args:
    - dataset_info: Named tuple. An object containing metadata of GQN datasets
    - mode: String. Can be 'train' or 'test'
    - root: String. Root directory of datasets

    Returns: List of files in the dataset
    """
    basepath = dataset_info.basepath
    base = os.path.join(root, basepath, mode)

    # usually of form '{}-of-{}.tfrecord'
    files = sorted(os.listdir(base))

    return [os.path.join(base, file) for file in files]

def _process_frames(dataset_info, example):
    """
    Obtain frame data from serialized representation

    Args:
    - dataset_info: Named tuple. An object containing metadata of GQN datasets
    - example: Serialized TFRecord object.

    Returns: A UInt8 Tensor of shape (B, S, W, H, C)
    - B: Batch size
    - S: Sequence size
    - W: Image width
    - H: Image height
    - C: Number of channels
    """
    frames = tf.concat(example['frames'], axis=0)
    frames = tf.map_fn(_convert_frame_data, tf.reshape(frames, [-1]), dtype=tf.float32, back_prop=False)
    img_dims = (dataset_info.frame_size, dataset_info.frame_size, 3)
    frames = tf.reshape(frames, (-1, dataset_info.sequence_size) + img_dims)

    if (dataset_info.frame_size != 64):
        frames = tf.reshape(frames, (-1, ) + img_dims)    # (B * S, W, H, C)
        default_img_dims = (64, 64, 3)
        frames = tf.image.resize_bilinear(frames, default_img_dims[:2], align_corners=True)
        frames = tf.reshape(frames, (-1, dataset_info.sequence_size) + default_img_dims)
    
    return frames

def _process_cameras(dataset_info, example, is_raw):
    """
    Obtain camera (in/extrinsic) data from serialized representation

    Args:
    - dataset_info: Named tuple. An object containing metadata of GQN datasets
    - example: Serialized TFRecord object.
    - is_raw: Boolean. If True, return raw camera data. Otherwise, process it

    Returns:
    - (is_raw = True) A Tensor of shape (B, S, 5)
    - (is_raw = False) A Tensor of shape (B, S, 7)
    """
    raw_cameras = example['cameras']
    raw_cameras = tf.reshape(raw_cameras, (-1, dataset_info.sequence_size, _NUM_POSE_PARAMS))

    if not is_raw:
        position = raw_cameras[:, :, 0:3]
        yaw = raw_cameras[:, :, 3:4]
        pitch = raw_cameras[:, :, 4:5]
        cameras = tf.concat([position, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
        return cameras
        
    else:
        return raw_cameras
        
def _make_context(frames, cameras):
    """
    Generate Context named tuple using camera, frame information

    Args:
    - cameras:
    - frames:

    Returns: A Context named tuple encapsulating given information
    """
    return Context(cameras=cameras, frames=frames)