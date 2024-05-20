import numbers
import os
import queue as Queue
import threading
from typing import Sequence

import torch
import mxnet as mx
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from .genDataSet import (
    TrainDataSplitMode,
    identifyTrainDataSet,
    testDataSet,
    quantTrainDataSet,
    mergeDataSet
)


# class BackgroundGenerator(threading.Thread):
#     def __init__(self, generator, local_rank, max_prefetch=6) -> None:
#         super(BackgroundGenerator, self).__init__()
#         self.queue = Queue.Queue(max_prefetch)
#         self.generator = generator
#         self.local_rank = local_rank
#         self.daemon = True
#         self.start()

#     def run(self):
#         torch.cuda.set_device(self.local_rank)
#         for item in self.generator:
#             self.queue.put(item)
#         self.queue.put(None)

#     def next(self):
#         next_item = self.queue.get()
#         if next_item is None:
#             raise StopIteration
#         else:
#             return next_item

#     def __next__(self):
#         return self.next()

#     def __iter__(self):
#         return self


# class DataLoaderX(DataLoader):
#     def __init__(self, local_rank, **kwargs):
#         super(DataLoaderX, self).__init__(**kwargs)
#         self.stream = torch.cuda.Stream(local_rank)
#         self.local_rank = local_rank

#     def __iter__(self):
#         self.iter = super(DataLoaderX, self).__iter__()
#         self.iter = BackgroundGenerator(self.iter, self.local_rank)
#         self.preload()
#         return self

#     def preload(self):
#         self.batch = next(self.iter, None)
#         if self.batch is None:
#             return None
#         with torch.cuda.stream(self.stream):
#             for k in range(len(self.batch)):
#                 self.batch[k] = self.batch[k].to(
#                     device=self.local_rank, non_blocking=True)

#     def __next__(self):
#         torch.cuda.current_stream(self.local_rank).wait_stream(self.stream)
#         batch = self.batch
#         if batch is None:
#             raise StopIteration
#         self.preload()
#         return batch


def my_collate_fn(batch):
    """
        假设 dataloader 是一个已创建的 DataLoader 对象

        自定义 collate_fn 函数用于数据加载时的处理

        避免出现 64 位的数据
    """
    collated_info = np.stack(
        [np.array(sample[1], dtype=object) for sample in batch])
    # 创建一个空字典来存储转换后的数据
    collated_batch = {}
    # 遍历每个样本的字典
    for key in batch[0][0].keys():
        # 将每个特征的数据类型转换为 float32
        collated_batch[key] = torch.stack([sample[0][key] for sample in batch])

    return collated_batch, collated_info


def dataLoader(
    batch_size: int,
    data_set,
    is_shuffle: bool = True,
    num_workers: int = 0
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
        # collate_fn=my_collate_fn,
        # prefetch_factor=2,
        # pin_memory=True,
        # persistent_workers=True
    )


def identifyTrainDataLoader(
    t_paths: Sequence[str],
    d_paths: Sequence[str],
    batch_size: int,
    features: Sequence[str],
    length_val_dataset: int,
    decoylength_val_dataset: int,
    traindata_splitmode: TrainDataSplitMode,
):
    train_data, val_data = identifyTrainDataSet(
        t_paths=t_paths,
        d_paths=d_paths,
        features=features,
        length_val_dataset=length_val_dataset,
        decoylength_val_dataset=decoylength_val_dataset,
        traindata_splitmode=traindata_splitmode
    )

    return dataLoader(batch_size, train_data), dataLoader(batch_size, val_data)


def quantTrainDataLoader(
    t_paths: Sequence[str],
    features: Sequence[str],
    length_val_dataset: int,
    batch_size: int,
):
    train_data, val_data = quantTrainDataSet(
        t_paths=t_paths,
        features=features,
        length_val_dataset=length_val_dataset
    )

    return dataLoader(batch_size, train_data), dataLoader(batch_size, val_data)


def testDataLoader(
    t_paths: Sequence[str],
    features: Sequence[str],
    batch_size: int,
):
    data = testDataSet(t_paths, features)
    return dataLoader(batch_size, data, False)


def multiTrainDataLoader(
    batch_size: int,
    t_paths_sequence: Sequence[Sequence[str]],
    d_paths_sequence: Sequence[Sequence[str]],
    features: Sequence[str],
    length_val_dataset: int,
    decoylength_val_dataset: int,
    traindata_splitmode: TrainDataSplitMode,
):
    train_data, val_data = None, None
    for t_paths, d_paths in zip(t_paths_sequence, d_paths_sequence):
        t, v = identifyTrainDataLoader(
            t_paths=t_paths,
            d_paths=d_paths,
            features=features,
            length_val_dataset=length_val_dataset,
            decoylength_val_dataset=decoylength_val_dataset,
            traindata_splitmode=traindata_splitmode,
            batch_size=batch_size
        )
        if train_data is None:
            train_data, val_data = t, v
        else:
            train_data, val_data = mergeDataSet(
                train_data, t), mergeDataSet(val_data, v)

    return dataLoader(batch_size, train_data), dataLoader(batch_size, val_data)
