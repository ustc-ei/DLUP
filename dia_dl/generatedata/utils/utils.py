import numpy as np

import os
import json
from typing import Literal, Dict


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

def save_json(path: str, content: Dict):
    with open(path, mode='w') as f:
        json.dump(content, f, ensure_ascii=False)


def is_data_exists(content, data_name, path: str):
    """
        判断数据是否准备就绪

        如果某数据已经准备就绪即其完成了数据提取以及峰合并操作，则会存在 json 中的 'dir_names' 中

        否则引发 ValueError
    """
    if data_name not in content['dir_names']:
        raise ValueError(f'{data_name} 未完成数据提取/峰合并操作')
    create_dir(path)


def read_label_result_path(content, data_name: str, type: Literal['label', 'result'], file_name: str):
    root_path = content['root']
    path = os.path.join(root_path, type, data_name, file_name)
    is_data_exists(content, data_name, os.path.dirname(path))
    return path


def read_dict_npy(path: str):
    return np.load(path, allow_pickle=True).item()


def read_npy(path: str):
    return np.load(path, allow_pickle=True)