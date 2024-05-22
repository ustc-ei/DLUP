from typing import Dict, Set, List

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd


def sort_protein_in_group(protein_group: str):
    """
        将蛋白质组中的蛋白质进行排序, 方便后面筛选

    Parameters:
    ---
    -   protein_group: 蛋白质组

    Returns:
    ---
    -   str: 排序之后的蛋白质组
    """
    proteins = protein_group.split(';')
    proteins.sort()
    sorted_protein_group = ';'.join(proteins)
    return sorted_protein_group


def normlize_uniprotID(uniport_id: str):
    """
        由于 proteinatlas.tsv 浓度表提供的 Uniport 列蛋白质组中蛋白质的分隔符为 " ," 和图谱库中的分隔符不一致, 因此需要规范化为图谱库的格式

        Parameters:
        ---
        -   uniport_id: 浓度表中的 uniprot_id

        Returns:
        ---
        -   str: 规范化后的蛋白质组
    """
    split_ids = uniport_id.split(", ")
    normlized_id = ";".join(split_ids)
    return sort_protein_in_group(normlized_id)


def generate_uniProtIds_bloodIm_bloodBm_map(proteinatlas: pd.DataFrame):
    uniprot_to_bloodIm_and_bloodMs_map = {}
    for uniprot, blood_im, blood_ms in proteinatlas[['Uniprot', 'Blood concentration - Conc. blood IM [pg/L]', 'Blood concentration - Conc. blood MS [pg/L]']].values:
        if not pd.isnull(uniprot):
            uniprot_to_bloodIm_and_bloodMs_map[normlize_uniprotID(uniprot)] = {
                'im': blood_im,
                'ms': blood_ms
            }
    return uniprot_to_bloodIm_and_bloodMs_map


def map_the_bloodIm_bloodMs_value(uniProtIds_bloodIm_bloodBm_map: Dict[str, Dict[str, np.float32]], protein_cup: Set):
    blood_im = []
    # blood_im_protein = []
    blood_ms = []
    # blood_ms_protin = []
    for protein in protein_cup:
        if protein in uniProtIds_bloodIm_bloodBm_map.keys():
            im = uniProtIds_bloodIm_bloodBm_map[protein]['im']
            ms = uniProtIds_bloodIm_bloodBm_map[protein]['ms']
            if not np.isnan(im):
                blood_im.append(im)
                # blood_im_protein.append(protein)
            if not np.isnan(ms):
                blood_ms.append(ms)
                # blood_ms_protin.append(protein)
    return blood_im, blood_ms


def calculate_cup(fileto_peptide_protein: Dict[str, Set[str]]):
    s: Set[str] = set()
    for value in fileto_peptide_protein.values():
        s = s | value
    return s


def generate_final_protein_result(path: str):
    protein_result = np.load(path, allow_pickle=True).item()
    return calculate_cup(protein_result)


def plot_protein_scatter(value, data_filenames: List[str], save_path: str, y_label: str, fig_title: str, is_train: bool, is_all_cup: bool):
    if is_train and not is_all_cup:
        fig_size = (12, 10)
    elif is_train and is_all_cup:
        fig_size = (6, 5)
    elif not is_train and is_all_cup:
        fig_size = (6, 5)
    else:
        fig_size = (8, 5)

    plt.rcParams['font.family'] = ["Times New Roman"]
    fig = plt.figure(figsize=fig_size, dpi=300)
    text = fig.suptitle(fig_title)
    text.set_fontsize(20)

    for i in range(len(value)):
        if is_all_cup:
            ax = plt.subplot(111)
            if len(value[0]) > 200:
                ax.set_xticks([0, 500, 1000, 1500])
            else:
                ax.set_xticks([0, 50, 100, 150])
        else:
            if is_train:
                ax = plt.subplot(2, 3, i + 1)
            else:
                ax = plt.subplot(1, 2, i + 1)
            ax.set_title(data_filenames[i])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.grid(True, 'major', 'y', **{
                'color': 'gray',
                'linewidth': 1.0,
                'linestyle': (1, (5, 1, 1, 1)),
                'alpha': 0.5
                })
        ax.set_facecolor('white')
        value_i = np.array(list(value[i]))  # type: ignore
        value_i = np.log(value_i) / np.log(10)
        x = np.linspace(0, len(value_i), len(value_i))
        value_sort_index = np.argsort(-value_i)
        value_sort = value_i[value_sort_index]
        ax.scatter(x, value_sort, s=20, color='#619DFF', alpha=0.8)
        ax.set_xlabel('Protein rank')  # type: ignore
        ax.set_ylabel(y_label + ' log$_{10}$ pg/L')

    fig.savefig(save_path + '.svg', dpi=800)


def generate_protein_info(library_path: str, proteinatlas_path: str, is_train: bool):
    sn_root = "./protein_result/spectronaut/"
    dl_root = "./protein_result/deeplearning/"
    s = "dataset"
    e = ".npy"
    train_dataset_n = list(range(1, 7))
    test_dataset_n = list(range(7, 10))
    train_dataset = [s + str(i) + e for i in train_dataset_n]
    test_dataset = [s + str(i) + e for i in test_dataset_n]
    library = pd.read_csv(library_path, sep='\t')
    proteinatlas = pd.read_csv(proteinatlas_path, sep='\t')
    uniProtIds_bloodIm_bloodBm_map = generate_uniProtIds_bloodIm_bloodBm_map(
        proteinatlas)

    sn_im_list = []
    sn_ms_list = []
    dl_im_list = []
    dl_ms_list = []
    sn_protein_all_result = set()
    dl_protein_all_result = set()

    datasets = train_dataset
    if not is_train:
        datasets = test_dataset

    for dataset in datasets:
        sn_dataset_path = sn_root + dataset
        dl_dataset_path = dl_root + dataset
        sn_protein_result = generate_final_protein_result(sn_dataset_path)
        dl_protein_result = generate_final_protein_result(dl_dataset_path)
        sn_im, sn_ms = map_the_bloodIm_bloodMs_value(
            uniProtIds_bloodIm_bloodBm_map, sn_protein_result)
        dl_im, dl_ms = map_the_bloodIm_bloodMs_value(
            uniProtIds_bloodIm_bloodBm_map, dl_protein_result)
        sn_im_list.append(sn_im)
        sn_ms_list.append(sn_ms)
        dl_im_list.append(dl_im)
        dl_ms_list.append(dl_ms)
        sn_protein_all_result = sn_protein_all_result | sn_protein_result
        dl_protein_all_result = dl_protein_all_result | dl_protein_result

    sn_im_result = np.array(sn_im_list, dtype=np.object_)
    sn_ms_result = np.array(sn_ms_list, dtype=np.object_)
    dl_im_result = np.array(dl_im_list, dtype=np.object_)
    dl_ms_result = np.array(dl_ms_list, dtype=np.object_)
    if is_train:
        np.save('./im/sn/train_dataset.npy', sn_im_result)
        np.save('./ms/sn/train_dataset.npy', sn_ms_result)
        np.save('./im/dl/train_dataset.npy', dl_im_result)
        np.save('./ms/dl/train_dataset.npy', dl_ms_result)
    else:
        np.save('./im/sn/test_dataset.npy', sn_im_result)
        np.save('./ms/sn/test_dataset.npy', sn_ms_result)
        np.save('./im/dl/test_dataset.npy', dl_im_result)
        np.save('./ms/dl/test_dataset.npy', dl_ms_result)

    sn_im, sn_ms = map_the_bloodIm_bloodMs_value(
        uniProtIds_bloodIm_bloodBm_map, sn_protein_all_result)
    dl_im, dl_ms = map_the_bloodIm_bloodMs_value(
        uniProtIds_bloodIm_bloodBm_map, dl_protein_all_result)

    sn_im_result = np.array(sn_im_list, dtype=np.object_)
    sn_ms_result = np.array(sn_ms_list, dtype=np.object_)
    dl_im_result = np.array(dl_im_list, dtype=np.object_)
    dl_ms_result = np.array(dl_ms_list, dtype=np.object_)

    if is_train:
        np.save('./im/sn/all_train_dataset.npy', sn_im_result)
        np.save('./ms/sn/all_train_dataset.npy', sn_ms_result)
        np.save('./im/dl/all_train_dataset.npy', dl_im_result)
        np.save('./ms/dl/all_train_dataset.npy', dl_ms_result)
    else:
        np.save('./im/sn/all_test_dataset.npy', sn_im_result)
        np.save('./ms/sn/all_test_dataset.npy', sn_ms_result)
        np.save('./im/dl/all_test_dataset.npy', dl_im_result)
        np.save('./ms/dl/all_test_dataset.npy', dl_ms_result)


def plot_protein_info_scatter_sn_dl_contrast(sn_result, dl_result, data_filenames: List[str], save_path: str, y_label: str, fig_title: str):
    plt.rcParams['font.family'] = ["Times New Roman"]
    fig = plt.figure(figsize=(15, 7), dpi=300)
    text = fig.suptitle(fig_title)
    text.set_fontsize('xx-large')

    y_label_text = fig.text(0.01, 0.25, y_label + ' log$_{10}$ pg/L')
    y_label_text.set_rotation('vertical')
    y_label_text.set_fontsize(12)
    for i in range(len(sn_result)):
        ax = plt.subplot(1, 3, i + 1)
        ax.grid(True, 'major', 'y', **{
            'color': 'gray',
            'alpha': 0.5,
            'linewidth': 1.0,
            'linestyle': (1, (4, 1, 1, 1))
        })
        ax_title = ax.set_title(data_filenames[i])
        ax_title.set_fontsize(18)
        ax.set_xlim(-1700, 1700)
        ax.set_ylim(3, 12)
        ax.set_yticks([4, 8, 12])

        custom_xticks = [-1500, -500, 0, 500, 1500]  # 新的刻度位置
        custom_xlabels = ['1500', '500', '0', '500', '1500']  # 对应的标签文本

        ax.set_xticks(custom_xticks)
        ax.set_xticklabels(custom_xlabels)

        # 将左侧的y轴和底部的x轴移动到中心位置
        ax.spines['left'].set_position('center')  # type: ignore

        # 隐藏右侧和顶部的轴
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # 设置轴标签
        ax.set_xlabel('Protein Rank', fontsize=12)
        # ax.set_ylabel(y_label + ' log_$_{10}$ pg/L')

        sn_ms, dl_ms = sn_result[i], dl_result[i]
        x_sn, x_dl = np.linspace(
            1, len(sn_ms) + 1, len(sn_ms)), -np.linspace(1, len(dl_ms) + 1, len(dl_ms))
        sn_ms, dl_ms = np.array(list(sn_ms)), np.array(
            list(dl_ms))  # type: ignore
        sn_min_argindex, dl_min_argindex = np.argsort(
            -sn_ms), np.argsort(-dl_ms)
        sn_ms, dl_ms = sn_ms[sn_min_argindex], dl_ms[dl_min_argindex]
        sn_ms, dl_ms = np.log(sn_ms) / np.log(10), np.log(dl_ms) / np.log(10)

        ax.scatter(x_dl, dl_ms, label='DeepLearning',
                   s=10, color='#BEB8DA', alpha=1.0)
        print(len(x_dl), len(x_sn))
        ax.text(-1000, 8, str(len(x_dl)), fontsize=18, color='#BEB8DA')
        ax.scatter(x_sn, sn_ms, label="Spectronaut",
                   s=10, color='#FA7F6F', alpha=1.0)
        ax.text(500, 8, str(len(x_sn)), fontsize=18, color='#FA7F6F')
        ax.legend(loc="upper right")
    fig.tight_layout()
    # 显示图像
    fig.savefig(save_path + '.svg', dpi=1000)


if __name__ == "__main__":
    library_path = './library/AD8-300S-directDIA.tsv'
    proteinatlas_path = './proteinatlas.tsv'
    generate_protein_info(library_path, proteinatlas_path, True)
    generate_protein_info(library_path, proteinatlas_path, False)
    sn_result, dl_result = np.load(
        "./ms/sn/test_dataset.npy", allow_pickle=True), np.load("./ms/dl/test_dataset.npy", allow_pickle=True)
    data_filenames = ['dataset7', 'dataset8', 'dataset9']
    save_path = 'test_dataset_dl_sn_contrast'
    plot_protein_info_scatter_sn_dl_contrast(sn_result, dl_result, data_filenames, save_path,
                                             'Blood concentration - Conc. blood MS', 'DeepLearning Spectronaut Contrast')
