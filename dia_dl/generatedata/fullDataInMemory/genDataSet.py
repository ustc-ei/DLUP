from enum import Enum
from typing import Any, Sequence, Tuple, Union, Dict

from torch.utils.data import Dataset, random_split, ConcatDataset, Subset


class DIADataSet(Dataset):
    def __init__(
        self,
        data: Dict,
        features: Sequence[str] = ['Spectrum', 'matchedMs2', 'label', 'info']
    ) -> None:
        """
            DIA 数据集类

            Parameters:
            ---
            -   data: 数据
            -   features: 采用的特征
        """
        super(DIADataSet, self).__init__()
        self.data = data
        self.features = features
        self.length = len(self.data[features[0]])

    def __getitem__(self, index: int) -> Any:
        """
            每个字段处的长度相同, 因此只需要 sample[f][index] 取即可
        """
        sample = {}
        for f in self.features:
            sample[f] = self.data[f][index]
        return sample

    def __len__(self):
        return self.length


def splitTrainAndValDataSet(
    train_dataset: Union[ConcatDataset, Subset, DIADataSet],
    length_val_dataset: int
) -> Tuple[Union[ConcatDataset, Subset, DIADataSet], ...]:

    train_dataSet, val_dataset = random_split(
        train_dataset,
        [len(train_dataset) - length_val_dataset, length_val_dataset]
    )

    return train_dataSet, val_dataset


def dataSet(
    data: Dict,
    features: Sequence[str] = ['Spectrum', 'matchedMs2', 'label', 'info']
):
    return DIADataSet(
        data=data,
        features=features
    )


class TrainDataSplitMode(Enum):
    """
    -   `TRAIN_VALIDATE_SPLIT`: 正常的训练集/验证集划分
    -   `TARGET_DECOY_TRAIN_VALIDATE`: `正库训练集`、`正库验证集`、`反库训练集`、`反库验证集`划分方式
    """
    TRAIN_VALIDATE_SPLIT = 0
    TARGET_DECOY_TRAIN_VALIDATE = 1


def identifyTrainDataSet(
    target_data,
    decoy_data,
    features: Sequence[str],
    length_val_dataset: int,
    decoylength_val_dataset: int,
    traindata_splitmode: TrainDataSplitMode,
):
    target_dataset = dataSet(target_data, features)
    decoy_dataset = dataSet(decoy_data, features)

    if traindata_splitmode == TrainDataSplitMode.TRAIN_VALIDATE_SPLIT:
        merged_data_set = mergeDataSet(target_dataset, decoy_dataset)
        train_data, val_data = splitTrainAndValDataSet(
            merged_data_set, length_val_dataset)

    elif traindata_splitmode == TrainDataSplitMode.TARGET_DECOY_TRAIN_VALIDATE:
        decoy_train_data, decoy_val_data = splitTrainAndValDataSet(
            decoy_dataset, decoylength_val_dataset)
        target_train_data, target_val_data = splitTrainAndValDataSet(
            target_dataset, length_val_dataset)
        train_data = mergeDataSet(target_train_data, decoy_train_data)
        val_data = mergeDataSet(decoy_val_data, target_val_data)
    return train_data, val_data


def quantTrainDataSet(
    data: Dict,
    features: Sequence[str],
    length_val_dataset: int,
) -> Tuple[DIADataSet, DIADataSet]:
    dataset = dataSet(data, features)
    train_data, val_data = splitTrainAndValDataSet(
        dataset, length_val_dataset)

    return train_data, val_data  # type: ignore


def testDataSet(
    data: Dict,
    features: Sequence[str]
):
    dataset = dataSet(data, features)
    return dataset


def mergeDataSet(*args):
    """
    多个数据集进行合并
    """
    return ConcatDataset([*args])
