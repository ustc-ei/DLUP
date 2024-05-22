from typing import List, Sequence
from matplotlib import axes
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import seaborn as sns
import pandas as pd


def set_ax_attributes(ax: Axes, label_fontsize: int, tick_labelsize: int):
    ax.set_xlabel('Dataset', fontsize=label_fontsize)
    ax.set_ylabel('Amount', fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=tick_labelsize)  # 设置x轴刻度标签的字体大小
    ax.tick_params(axis='y', labelsize=tick_labelsize)  # 设置y轴刻度标签的字体大小
    ax.tick_params(axis='x', rotation=45)


def plot_box(result: pd.DataFrame, dataset_names, title: str, is_train: bool):
    num = result[result['Methods'] == 'deeplearning'].groupby(
        'dataset_label').count()['value'].values
    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = ["Times New Roman"]  # type: ignore
    if is_train:
        figsize = (8, 8)
    else:
        figsize = (8, 8)
    figure = plt.figure(dpi=300, figsize=figsize)
    ax = plt.subplot(111)
    sns.boxplot(data=result, x='dataset_label', y='value',
                hue='Methods', palette=sns.color_palette('pastel'), ax=ax)
    fig_title = figure.suptitle('ProteinGroups_Num: ' + title)
    fig_title.set_fontsize(20)
    dataset_names_n = []
    for i, dataset in enumerate(dataset_names):
        dataset_names_n.append(dataset + f"\n({num[i]})")
    sns.swarmplot(data=result, x='dataset_label', y='value', hue='Methods', dodge=True,
                  size=3, ax=ax, palette=sns.color_palette('deep'), legend=False)  # type: ignore
    ax.set_xticks([i for i in range(0, len(dataset_names))], dataset_names_n)
    set_ax_attributes(ax, 18, 14)
    ax.legend(bbox_to_anchor=(1.3, 0.1))
    figure.tight_layout()
    plt.savefig(f"./{title}.svg", dpi=1000)


def load_protein_result(files: List[str]):
    dataset_names = []
    dataset_proteinLens = []
    for f in files:
        file_proteinSet = np.load(f, allow_pickle=True).item()
        dataset_name = f.split('/')[-1].split('.')[0]
        file_proteinLen = {
            k: len(val)
            for k, val in file_proteinSet.items()
        }
        proteinLen_list = list(file_proteinLen.values())
        dataset_proteinLens.append(proteinLen_list)
        dataset_names.append(dataset_name)
    return dataset_proteinLens, dataset_names


def insert_to_df_column(methods, dataset, values, labels: List[str], result: List[List[int]], method: str):
    for i, items in enumerate(result):
        for item in items:
            values.append(item)
            dataset.append(int(labels[i].split('t')[-1]))
            methods.append(method)


def generate_dataset_df(dl_dataset_proteinLens, sn_dataset_proteinLens, dataset_names):
    methods = []
    dataset = []
    values = []
    insert_to_df_column(methods, dataset, values, dataset_names,
                        dl_dataset_proteinLens, 'deeplearning')
    insert_to_df_column(methods, dataset, values, dataset_names,
                        sn_dataset_proteinLens, 'spectronaut')
    df = pd.DataFrame(columns=['Methods', 'dataset_label', 'value'])
    df['Methods'] = methods
    df['dataset_label'] = dataset
    df['value'] = values
    return df


def plot(dl_datasets, sn_datasets, title: str, is_train: bool):
    dl, dataset_names = load_protein_result(dl_datasets)
    sn, _ = load_protein_result(sn_datasets)
    df = generate_dataset_df(dl, sn, dataset_names)
    plot_box(df, dataset_names, title, is_train)


if __name__ == "__main__":
    dl_root_path = "./protein_result/deeplearning/"
    sn_root_path = "./protein_result/spectronaut/"
    s, e = "dataset", ".npy"

    train_dataset_n, test_dataset_n = list(range(1, 7)), list(range(7, 10))
    train_dataset, test_dataset = [
        s + str(i) + e for i in train_dataset_n], [s + str(i) + e for i in test_dataset_n]

    dl_train, dl_test = [
        dl_root_path + f for f in train_dataset], [dl_root_path + f for f in test_dataset]
    sn_train, sn_test = [
        sn_root_path + f for f in train_dataset], [sn_root_path + f for f in test_dataset]

    plot(dl_train, sn_train, 'train_result', True)
    plot(dl_test, sn_test, 'test_result', False)
