"""
Converts selected data to Pytorch compatible formats (e.g. Numpy arrays, etc...)
"""

import os
# disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE" 
import sys
from pathlib import Path
from multiprocessing import Process
from tqdm import tqdm

# constants
BASE_DIR = os.path.join(Path(__file__).parent.absolute(), '../')    # path to project root directory

sys.path.append(BASE_DIR)    # append project root to import paths

from utils.tfrecord_converter import _DATASETS
from utils.tfrecord_converter import *

# suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

    cnt = 0
    total = 1000000

    pbar = tqdm(total=total, leave=False)

    for file in files:
        engine = tf.python_io.tf_record_iterator(file)

        for i, raw_data in enumerate(engine):
            converted_file = os.path.join(converted_dataset_train, f'{cnt+i}.p')
            # print(f' [-] converting scene {file}-{i} into {converted_file}')
            p = Process(target=convert_raw_to_numpy, args=(dataset_info, raw_data, converted_file))
            p.start()
            p.join()

        cnt += i
        pbar.update(cnt)

        # restrict the number of train data samples -> 1 M
        if cnt >= total:
            print(f' [-] Converted total {cnt+1} contexts in the training set')
            break

    pbar.close()

    # convert test data
    files = get_dataset_files(dataset_info, 'test', '.')

    cnt = 0
    total = 200000

    pbar = tqdm(total=total, leave=False)

    for file in files:
        engine = tf.python_io.tf_record_iterator(file)
        for i, raw_data in enumerate(engine):
            converted_file = os.path.join(converted_dataset_test, f'{cnt+i}.p')
            # print(f' [-] converting scene {file}-{i} into {converted_file}')
            p = Process(target=convert_raw_to_numpy, args=(dataset_info, raw_data, converted_file))
            p.start()
            p.join()

        cnt += i
        pbar.update(cnt)

        # restrict the number of data samples -> 200 Thousand
        if cnt >= total:
            print(f' [-] Converted total {cnt+1} contexts in the test set')
            break

    pbar.close()


if __name__ == '__main__':
    main()