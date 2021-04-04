"""
Converts selected data to Pytorch compatible formats (e.g. Numpy arrays, etc...)

TODO: Implement this when necessary
"""

if __name__ == '__main__':

    import sys

    if len(sys.argv) < 3:
        print('[!] Please specify a dataset to convert')
        print('[!] Command is of form: python convert_to_torch.py dataset_name train/test/all')
        print('[!] e.g. python convert_to_torch.py jaco train')
        exit()

    DATASET = sys.argv[1]
    dataset_info = _DATASETS[DATASET]

    converted_dataset_path = f'{DATASET}-torch'

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