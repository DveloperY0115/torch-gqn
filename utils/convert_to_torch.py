"""
Script for converting TFRecord format to Pytorch compatible formats
"""

import os
import sys
import tensorflow as tf

BASE_DIR = '../'    # path to project root directory
sys.path.append(BASE_DIR)

def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)