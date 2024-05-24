import os
import argparse
from typing import Dict

import torch
import numpy as np
import pandas as pd
import numpy.typing as npt

import read_tims_data
from process import peptide_quantification, protein_quantification

from dia_dl.generatedata.utils.utils import read_json, create_dir, read_dict_npy
from dia_dl.generatedata import genData
from dia_dl.generatedata.fullDataInMemory.genDataLoader import testDataLoader
from dia_dl.model.model import identification_model, quantification_model
from dia_dl.model.model_utils import classfier_test, regression_test
from dia_dl.model.utils import get_dl_identify_modified


def create_parent_dirs(root: str, data_name: str):
    target_path = os.path.join(
        root, 'identification', 'test', 'target', data_name)
    decoy_path = os.path.join(root, 'identification',
                              'test', 'decoy', data_name)
    quant_path = os.path.join(root, 'quantification', 'test', data_name)
    create_dir(target_path)
    create_dir(decoy_path)
    create_dir(quant_path)
    return target_path, decoy_path, quant_path


def load_data(
    root: str,
    data_name: str,
    identify_configs: Dict,
    quant_configs: Dict,
    batch_size: int
):
    collection_str = 'collection.npy'
    target_path, decoy_path, quant_path = create_parent_dirs(root, data_name)
    target_path = os.path.join(target_path, collection_str)
    decoy_path = os.path.join(decoy_path, collection_str)
    quant_path = os.path.join(quant_path, collection_str)

    # load data
    target = np.load(target_path, allow_pickle=True).item()
    target['matchedMs2'] = target['matchedMs2'] / np.max(target['matchedMs2'])

    decoy = np.load(decoy_path, allow_pickle=True).item()
    decoy['matchedMs2'] = decoy['matchedMs2'] / np.max(decoy['matchedMs2'])

    quant = np.load(quant_path, allow_pickle=True).item()

    target = testDataLoader(
        target, identify_configs['features']['read'], batch_size)
    decoy = testDataLoader(
        decoy, identify_configs['features']['read'], batch_size)
    quant = testDataLoader(
        quant, quant_configs['features']['read'], batch_size)

    return target, decoy, quant


def model_test(root, data_name):
    model_configs = read_json('./configs/model_configs.json')
    identify_configs, quant_configs = model_configs['identification'], model_configs['quantification']

    target, decoy, quant = load_data(
        root=root,
        data_name=data_name,
        identify_configs=identify_configs,
        quant_configs=quant_configs,
        batch_size=model_configs['batch_size']
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    identify_model = identification_model(6, 6).to(device)
    quant_model = quantification_model(6, 6).to(device)

    identify_modelstate = torch.load('./model/identification.ckpt')
    quant_modelstate = torch.load('./model/quantification.ckpt')

    identify_model.load_state_dict(identify_modelstate)
    quant_model.load_state_dict(quant_modelstate)

    targte_result = classfier_test(
        identify_model,
        device,
        target,
        identify_configs
    )

    decoy_result = classfier_test(
        identify_model,
        device,
        decoy,
        identify_configs
    )

    quant_result = regression_test(
        quant_model,
        device,
        quant,
        quant_configs
    )
    return targte_result, decoy_result, quant_result


def process_result(
    target_result: Dict[str, npt.NDArray],
    decoy_result: Dict[str, npt.NDArray],
    quant_result: Dict[str, npt.NDArray],
    target_library: Dict[str, Dict],
    root: str,
    data_name: str
):
    quantity, info = quant_result['quantity'], quant_result['info']
    indices = (quantity > 0).nonzero()
    quantity = quantity[indices]
    info_sequence = info[indices]
    files = set(info_sequence[:, 0])

    quant = {
        f: {}
        for f in files
    }

    identify_result = get_dl_identify_modified(
        t=target_result,
        d=decoy_result
    )

    for q, info in zip(quantity, info_sequence):
        file, modified_charge = info
        if modified_charge in identify_result[file]:
            quant[file][modified_charge] = {
                'quantity': q,
                'strippedSequence': target_library[modified_charge]['StrippedPeptide']
            }

    peptide_quantification.quantification(quant, os.path.join(
        root, f'{data_name}_peptide_quantification.tsv'))
    protein_quantification.quantification(quant, target_library, os.path.join(
        root, f'{data_name}_protein_quantification.tsv'))


def preprocess(root_path: str, num_processes: int = 20):
    # get the dir name
    data_name = root_path.split('/')[-1]
    preprocess_configs = read_json('./configs/preprocess_configs.json')
    # extract the raw data and merge peaks
    read_tims_data.main(
        root_path=root_path,
        num_processes=num_processes,
        tol=preprocess_configs['tol']
    )
    """
        preprocess the raw data
        identification
        ---
            use the Manhattan Distance distance to filter the most similar ms2
            features
            -   intensity
            -   ionmobility
        quantification  
        ---
            use the Manhattan Distance and the Peaks intensity sum to filter the most similar ms2
            features
            - intensity 
    """
    target_path = preprocess_configs['library']['target_path']
    decoy_path = preprocess_configs['library']['decoy_path']

    merge_path = os.path.join(root_path, 'merge')
    files = [os.path.join(merge_path, f)
             for f in os.listdir(merge_path) if f.endswith('.npy')]
    # identification
    preprocess_configs['is_test'] = True
    preprocess_configs['is_quant'] = False
    # just the root path
    save_root_path = preprocess_configs['save_root_path']
    preprocess_configs['num_processes'] = num_processes
    """
        then we will create the dirs for the identification and quantification

        the below is an example

        folder (the save_root_path)
        --- identification
            --- train (optional if you want to fune the model)
            (the dirs are similar to the **test**)
            --- test
                --- target
                    --- data_name
                        --- (modifiedpeptide, charge).npy
                --- decoy
                (the dirs are similar to the **target**)
        --- quantification
            --- train
            --- test
                --- target
                    --- data_name
                        --- (modifiedpeptide, charge).npy
    """
    target_dir, decoy_dir, quant_dir = create_parent_dirs(
        save_root_path, data_name)
    # target
    preprocess_configs['save_root_path'] = target_dir
    preprocess_configs['is_decoy'] = False
    genData.main(
        libraryPath=target_path,
        massSpectrumFilePathList=files,
        labels=None,
        configs=preprocess_configs
    )
    # decoy
    preprocess_configs['save_root_path'] = decoy_dir
    preprocess_configs['is_decoy'] = True
    genData.main(
        libraryPath=decoy_path,
        massSpectrumFilePathList=files,
        labels=None,
        configs=preprocess_configs
    )
    preprocess_configs['is_quant'] = True
    # quantification
    preprocess_configs['save_root_path'] = quant_dir
    genData.main(
        libraryPath=target_path,
        massSpectrumFilePathList=files,
        labels=None,
        configs=preprocess_configs
    )


def main(root: str):
    data_name = root.split('/')[-1]
    preprocess(root, 5)
    preprocess_configs = read_json('./configs/preprocess_configs.json')
    save_root_path = preprocess_configs['save_root_path']
    # model test
    target_result, decoy_result, quant_result = model_test(
        save_root_path, data_name)
    # process identification and quantification result
    target_library = read_dict_npy(
        preprocess_configs['library']['target_path'])
    process_result(target_result, decoy_result, quant_result,
                   target_library, save_root_path, data_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Please input data folder path and num of mutil process")

    # 定义字符串类型的可选参数
    parser.add_argument('--data_path', type=str, help='data foler path')

    # 定义整数类型的可选参数
    parser.add_argument('--num_process', type=int,
                        help='number of the process')

    # 解析命令行参数
    args = parser.parse_args()
    main(args.data_path, argparse.num_process)
