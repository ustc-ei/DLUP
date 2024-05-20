from typing import Sequence, Dict

import torch
import numpy as np
from torch.utils.data import DataLoader

from .genDataSet import (
    TrainDataSplitMode,
    identifyTrainDataSet,
    splitTrainAndValDataSet,
    testDataSet,
    quantTrainDataSet,
    mergeDataSet
)


def my_collate_fn(batch):
    """
        假设 dataloader 是一个已创建的 DataLoader 对象

        自定义 collate_fn 函数用于数据加载时的处理

        避免出现 64 位的数据
    """
    collated_info = np.stack(
        [np.array(sample['info'], dtype=object) for sample in batch])
    # 创建一个空字典来存储转换后的数据
    collated_batch = {}
    # 遍历每个样本的字典
    for key in batch[0].keys():
        if key == 'info':
            continue
        # 将每个特征的数据类型转换为 float32
        collated_batch[key] = torch.stack(
            [torch.tensor(sample[key]) for sample in batch])

    return collated_batch, collated_info


def dataLoader(
    batch_size: int,
    data_set,
    is_shuffle: bool = True,
    num_workers: int = 8
):
    """
        由输入的 `dataset` 生成对应的 `dataLoader`
    ### Input Parameters:
    -   `batchSize`: batch 大小
    -   `dataSet`: 数据集
    -   `isShuffle`: 是否随机化

    ### Return:
    -   对应的 `DataLoader`
    """
    return DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        collate_fn=my_collate_fn,
    )


def identifyTrainDataLoader(
    target_data: Dict,
    decoy_data: Dict,
    batch_size: int,
    features: Sequence[str],
    length_val_dataset: int,
    decoylength_val_dataset: int,
    traindata_splitmode: TrainDataSplitMode,
):
    train_data, val_data = identifyTrainDataSet(
        target_data=target_data,
        decoy_data=decoy_data,
        features=features,
        length_val_dataset=length_val_dataset,
        decoylength_val_dataset=decoylength_val_dataset,
        traindata_splitmode=traindata_splitmode
    )

    return dataLoader(batch_size, train_data), dataLoader(batch_size, val_data)


def quantTrainDataLoader(
    data: Dict,
    features: Sequence[str],
    length_val_dataset: int,
    batch_size: int,
):
    train_data, val_data = quantTrainDataSet(
        data=data,
        features=features,
        length_val_dataset=length_val_dataset
    )

    return dataLoader(batch_size, train_data), dataLoader(batch_size, val_data)


def testDataLoader(
    data: Dict,
    features: Sequence[str],
    batch_size: int,
):
    data = testDataSet(data, features)
    return dataLoader(batch_size, data, False)


def multiIdentifyTrainDataLoader(
    batch_size: int,
    target_data_sequence: Sequence[Dict],
    decoy_data_sequence: Sequence[Dict],
    features: Sequence[str],
    length_val_dataset: int,
    decoylength_val_dataset: int,
    traindata_splitmode: TrainDataSplitMode,
):
    train_data, val_data = None, None
    for target_data, decoy_data in zip(target_data_sequence, decoy_data_sequence):
        t, v = identifyTrainDataSet(
            target_data=target_data,
            decoy_data=decoy_data,
            features=features,
            length_val_dataset=length_val_dataset,
            decoylength_val_dataset=decoylength_val_dataset,
            traindata_splitmode=traindata_splitmode
        )
        if train_data is None:
            train_data, val_data = t, v
        else:
            train_data, val_data = mergeDataSet(
                train_data, t), mergeDataSet(val_data, v)

    return dataLoader(batch_size, train_data), dataLoader(batch_size, val_data)


def multiQuantTrainDataLoader(
    batch_size: int,
    data_sequence: Sequence[Dict],
    features: Sequence[str],
    length_val_dataset: int,
):
    train_data = None
    for data in data_sequence:
        t = testDataSet(
            data=data,
            features=features,
        )
        if train_data is None:
            train_data = t
        else:
            train_data = mergeDataSet(train_data, t)

    train_data, val_data = splitTrainAndValDataSet(
        train_dataset=train_data,
        length_val_dataset=length_val_dataset
    )

    return dataLoader(batch_size, train_data), dataLoader(batch_size, val_data)
