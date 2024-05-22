import numpy as np

import os
import json
<<<<<<< HEAD
from typing import Literal, Dict
=======
from typing import Literal
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79


def create_dir(path: str):
    """
        创建文件夹
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"create {path} success!")


def read_json(path: str):
    """
        读取 json 文件
    """
    with open(path) as f:
        content = json.loads(f.read())
    return content

<<<<<<< HEAD
def save_json(path: str, content: Dict):
    with open(path, mode='w') as f:
        json.dump(content, f, ensure_ascii=False)

=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79

def is_data_exists(content, data_name, path: str):
    """
        判断数据是否准备就绪

        如果某数据已经准备就绪即其完成了数据提取以及峰合并操作，则会存在 json 中的 'dir_names' 中

        否则引发 ValueError
    """
    if data_name not in content['dir_names']:
        raise ValueError(f'{data_name} 未完成数据提取/峰合并操作')
    create_dir(path)


<<<<<<< HEAD
=======
def read_data_task_path(content, data_name: str, task_type: Literal['identify', 'quant'], data_usage: Literal['train', 'test', 'temp'], file_name: str):
    """
        获取`定性/定量` `训练/测试/临时`数据的文件路径

        Parameters:
        ---
        -   content: 文件路径的 json 文件内容
        -   data_name: 数据名
        -   task_type: 定性/定量, 只能为 'identify'/'quant'
        -   data_usage: 训练/测试, 只能为 'train'/'test'/'temp'
        -   file_name: 读取的文件名, 一般是 .npy 为后缀的文件
    """
    root_path = content['root']
    train_test_data_folder = content['train_test_data_folder']
    path = os.path.join(root_path, train_test_data_folder,
                        task_type, data_name, data_usage, file_name)
    is_data_exists(content, data_name, os.path.dirname(path))
    return path


>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
def read_label_result_path(content, data_name: str, type: Literal['label', 'result'], file_name: str):
    root_path = content['root']
    path = os.path.join(root_path, type, data_name, file_name)
    is_data_exists(content, data_name, os.path.dirname(path))
    return path


def read_dict_npy(path: str):
    return np.load(path, allow_pickle=True).item()


def read_npy(path: str):
<<<<<<< HEAD
    return np.load(path, allow_pickle=True)
=======
    return np.load(path, allow_pickle=True)


def read_data_task_path_infile_module(
    content,
    data_name: str,
    task_type: Literal['identify', 'quant'],
    data_usage: Literal['train', 'test'],
    cv_str: str = 'cv0.1'
):
    """
        TODO: 
        将 is_train 参数删除，使用 data_usage 进行判定替换
        Parameters:
        ---
        -   content: 文件路径的 json 文件内容
        -   data_name: 数据名
        -   task_type: 定性/定量, 只能为 'identify'/'quant'
        -   data_usage: 训练/测试, 只能为 'train'/'test'
    """
    root_path = content['root']
    train_test_data_folder = content['train_test_data_folder']
    if data_usage == 'train':
        path = os.path.join(root_path, train_test_data_folder,
                            task_type, data_usage, cv_str, data_name)
    else:
        path = os.path.join(root_path, train_test_data_folder,
                            task_type, data_usage, data_name)
    folders = [os.path.join(path, f) for f in os.listdir(
        path) if os.path.isdir(os.path.join(path, f))]
    paths = [
        os.path.join(folder, f)
        for folder in folders
        for f in os.listdir(folder)
        if f != 'collection.npy'
    ]
    return paths
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
