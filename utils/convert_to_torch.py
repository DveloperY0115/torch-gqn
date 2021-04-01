"""
Script for converting TFRecord format to Pytorch compatible formats
"""

import os
import sys
from torch.utils.data import Dataset, DataLoader

import data_reader

BASE_DIR = '../'    # path to project root directory
sys.path.append(BASE_DIR)

