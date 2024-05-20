import os
from typing import Literal

from dia_dl.generatedata.genData import main, read_dict_npy
from dia_dl.generatedata.utils.utils import create_dir, read_json
import read_tims_data
import multi_process


def read_data(root_path: str, extension_class: Literal['.d', '.mzML'], num_processes: int = 20, tol: int = 15):
    if extension_class == '.d':
        read_tims_data.main(
            root_path=root_path,
            num_processes=num_processes,
            tol=tol
        )
    elif extension_class == '.mzML':
        multi_process.main(
            root_path=root_path,
            num_processes=num_processes,
            tol=tol
        )
    print('read data end!')


def generate_identify_train_data(data_name: str, cv: Literal['cv0.1', 'cv0.15', 'cv0.2']):
    pass


def generate_identify_test_data(data_name: str):
    pass


def generate_quant_train_data(data_name: str, cv: Literal['cv0.1', 'cv0.15', 'cv0.2']):
    configs['is_quant'] = True
    configs['is_test'] = False
    configs = read_json('./quant_configs.json')
    configs = configs[data_name]
    files = [os.path.join(configs['root_path'], f)
             for f in os.listdir(configs['root_path']) if f.endswith('.npy')]
    label_path = os.path.join(configs['label_path'], f'{cv}.npy')
    print('read labels')
    labels = read_dict_npy(label_path)
    print('end')
    root = '/data/xp/label_in_file/quant/train'
    configs['save_root_path'] = os.path.join(
        root, f'{cv}', data_name)
    create_dir(configs['save_root_path'])
    # quant target 测试集
    main(configs['library']['target_path'],
         files, labels, configs)  # type: ignore


def generate_quant_test_data(data_name: str):
    configs['is_test'] = True
    configs['is_quant'] = True
    configs = read_json('./quant_configs.json')
    configs = configs[data_name]
    files = [os.path.join(configs['root_path'], f)
             for f in os.listdir(configs['root_path']) if f.endswith('.npy')]

    labels = None
    save_root_path = os.path.join(
        '/data/xp/label_in_file/quant/test', data_name)
    configs['save_root_path'] = save_root_path
    create_dir(configs['save_root_path'])
    main(configs['library']['target_path'],
         files, labels, configs)  # type: ignore


def main(root_path: str, data_name: str, extension_class: Literal['.d', '.mzML'], num_processes: int = 20, tol: int = 15):
    read_data(
        root_path=root_path,
        extension_class=extension_class,
        num_processes=num_processes,
        tol=tol
    )

    generate_quant_test_data(data_name)
