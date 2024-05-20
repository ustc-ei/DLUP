from enum import Enum
from typing import Any, Sequence, Tuple, Union

from torch.utils.data import Dataset, random_split, ConcatDataset, Subset

from ..genData import read_dict_npy


class DIADataSetWithFilePath(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        features: Sequence[str] = ['Spectrum', 'matchedMs2', 'label']
    ) -> None:
        super(DIADataSetWithFilePath, self).__init__()
        self.paths = paths
        self.features = features
        self.length = len(self.paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> Any:
        path: str = self.paths[index]  # type: ignore
        data = read_dict_npy(path)
        # 获取(modified_peptide, charge)
        file = path.split('/')[-2]
        modified_charge = path.split('/')[-1].split('.npy')[0][1:-1]
        modified_charge = modified_charge.split(', ')
        modified, charge = modified_charge[0], modified_charge[1]
        pep_info = (file, (modified, int(charge)))

        sample = {}
        for f in self.features:
            sample[f] = data[f]
        # print(len(sample))
        return sample, pep_info


def splitTrainAndValDataSet(
    train_dataset: Union[ConcatDataset, Subset, DIADataSetWithFilePath],
    length_val_dataset: int
) -> Tuple[Union[ConcatDataset, Subset, DIADataSetWithFilePath], ...]:

    train_dataSet, val_dataset = random_split(
        train_dataset,
        [len(train_dataset) - length_val_dataset, length_val_dataset]
    )

    return train_dataSet, val_dataset


def dataSet(
    paths: Sequence[str],
    features: Sequence[str] = ['matchedMs2', 'Spectrum', 'label']
):
    return DIADataSetWithFilePath(
        paths=paths,
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
    t_paths: Sequence[str],
    d_paths: Sequence[str],
    features: Sequence[str],
    length_val_dataset: int,
    decoylength_val_dataset: int,
    traindata_splitmode: TrainDataSplitMode,
):
    target_dataset = dataSet(t_paths, features)
    decoy_dataset = dataSet(d_paths, features)

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
    t_paths: Sequence[str],
    features: Sequence[str],
    length_val_dataset: int,
) -> Tuple[DIADataSetWithFilePath, DIADataSetWithFilePath]:
    dataset = dataSet(t_paths, features)
    train_data, val_data = splitTrainAndValDataSet(
        dataset, length_val_dataset)

    return train_data, val_data  # type: ignore


def testDataSet(
    t_paths: Sequence[str],
    features: Sequence[str]
):
    dataset = dataSet(t_paths, features)
    return dataset


def mergeDataSet(*args):
    """
    多个数据集进行合并
    """
    return ConcatDataset([*args])
