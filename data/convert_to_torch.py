"""
Converts selected data to Pytorch compatible formats (e.g. Numpy arrays, etc...)
"""

import os
import sys
from pathlib import Path
from multiprocessing import Process

# constants
BASE_DIR = os.path.join(Path(__file__).parent.absolute(), '../')    # path to project root directory

sys.path.append(BASE_DIR)    # append project root to import paths

from utils.tfrecord_converter import _DATASETS
from utils.tfrecord_converter import *

def main():
    if len(sys.argv) < 2:
        print('[!] Please specify a dataset to convert')
        print('[!] Command is of form: python convert_to_torch.py dataset_name')
        print('[!] e.g. python convert_to_torch.py jaco')
        exit(-1)

    DATASET = sys.argv[1]

    dataset_info = _DATASETS[DATASET]

    converted_dataset_path = f'{DATASET}_torch'
    converted_dataset_train = f'{converted_dataset_path}/train'
    converted_dataset_test = f'{converted_dataset_path}/test'

    if not os.path.exists(converted_dataset_path):
        os.mkdir(converted_dataset_path)
    if not os.path.exists(converted_dataset_train):
        os.mkdir(converted_dataset_train)
    if not os.path.exists(converted_dataset_test):
        os.mkdir(converted_dataset_test)

    # convert training data
    files = get_dataset_files(dataset_info, 'train', '.')

    tot = 0
    for file in files:
        engine = tf.python_io.tf_record_iterator(file)
        for i, raw_data in enumerate(engine):
            converted_file = os.path.join(converted_dataset_train, f'{tot+i}.p')
            print(f' [-] converting scene {file}-{i} into {converted_file}')
            p = Process(target=convert_raw_to_numpy, args=(dataset_info, raw_data, converted_file))
            p.start()
            p.join()
        tot += i

    print(f' [-] Converted total {tot} contexts in the training set')

    # convert test data
    files = get_dataset_files(dataset_info, 'test', '.')

    tot = 0
    for file in files:
        engine = tf.python_io.tf_record_iterator(file)
        for i, raw_data in enumerate(engine):
            converted_file = os.path.join(converted_dataset_test, f'{tot+i}.p')
            print(f' [-] converting scene {file}-{i} into {converted_file}')
            p = Process(target=convert_raw_to_numpy, args=(dataset_info, raw_data, converted_file))
            p.start()
            p.join()
        tot += i

    print(f' [-] Converted total {tot} contexts in the test set')

if __name__ == '__main__':
    main()