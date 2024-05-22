from ast import Call
from typing import Dict, Set, List, Callable, Sequence, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib_venn import venn2, venn2_circles
from matplotlib.text import Text


def plot_venn_ax(left: int, right: int, overlap: int, ax: Axes, ax_title: str):
    """
        绘制 venn 图

        Parameters:
        ---
        -   left: 左侧的值
        -   right: 右侧的值
        -   overlap: 相交部分的值
        -   ax: 处于哪个绘图区域
        -   ax_title: ax 的标题
    """
    ax.set_title(ax_title)
    v = venn2(
        subsets=(left - overlap,  right - overlap, overlap),
        set_labels=[f'{round(overlap / left * 100, 1)}%', ''],
        normalize_to=1.0,
        set_colors=['#BEB8DA', '#FA7F6F'],
        alpha=0.6,
        ax=ax
    )
    label: Text = v.set_labels[0]  # type: ignore
    x, y = label.get_position()
    label.set_position((x + 0.15, y))
    label.set_fontsize(15)
    label.set_color('#BEB8DA')
    label.set_alpha(1.0)
    label.set_fontweight('bold')
    venn2_circles(
        subsets=(left - overlap,  right - overlap, overlap),
        linestyle=(1, (5, 1, 1, 2)),  # type: ignore
        linewidth=2.0,
        color='gray',
        alpha=0.5,
        ax=ax
    )
    ax.legend(['DeepLearning', 'Spectronaut'], loc='upper right')


def plot_venn_figure(deep_learning: List, spectronaut: List, overlaps: List, data_filename: List[str], is_train: bool, fig_title: str, save_rootpath: str):
    plt.rcParams['font.family'] = ["Times New Roman"]

    plt.style.use('bmh')
    if is_train:
        fig = plt.figure(figsize=(10, 8), dpi=300)
    else:
        fig = plt.figure(figsize=(10, 5), dpi=300)
    text = fig.suptitle(fig_title)
    text.set_fontsize(20)

    for i in range(len(deep_learning)):
        if is_train:
            ax = plt.subplot(2, 3, i + 1)
        else:
            ax = plt.subplot(1, 3, i + 1)
        plot_venn_ax(deep_learning[i], spectronaut[i],
                     overlaps[i], ax, data_filename[i])
    fig.tight_layout()
    fig.savefig(f"{save_rootpath + fig_title.split(' ')[-1]}.svg", dpi=1000)


def calculate_cup(d: Dict[str, Set[str]]):
    """
        计算各重复样本结果的并集
    """
    files = list(d.keys())
    initial_file = files[0]
    cup = d[initial_file].copy()
    for f in files[1:]:
        cup |= d[f]
    return cup


def calculate_overlap(d: Dict[str, Set[str]], overlap_func: Callable[[Set[str], Set[str]], Set[str]]):
    """
        计算各重复样本的 overlap

        Parameters:
        ---
        -   d: key 为文件名, value 为该文件下的结果
        -   overlap_func: 计算 overlap 的函数, peptide 和 protein_groups 的计算方式不一样

        Returns:
        ---
        -   set: 所有重复样本的 overlap
    """
    files = list(d.keys())
    initial_file = files[0]
    cap = d[initial_file].copy()
    for _, f in enumerate(files[1:]):
        cap = overlap_func(cap, d[f])
    return cap


def calculate_dl_sn_overlap(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]], overlap_func: Callable[[Set[str], Set[str]], Set[str]]):
    """
        计算 dl, sn 两种方法在各重复样本下的 overlap
    """
    overlap: Dict[str, Set[str]] = {}
    for file in dl.keys():
        overlap[file] = overlap_func(dl[file], sn[file])
    return overlap


def calculate_average(d: Dict[str, Set[str]]):
    """
        计算各重复样本下的鉴定数量的平均值
    """
    d_num = {key: len(value) for key, value in d.items()}
    average = int(sum(d_num.values()) / len(d_num.keys()))
    return average


def load_dict_data(path: str):
    return np.load(path, allow_pickle=True).item()


def get_fig_title(is_train: bool, is_peptide: bool):
    peptide, protein_groups = 'Peptide: ', 'ProteinGroups: '
    train, test = 'train_dataset_result', 'test_dataset_result'
    if is_train:
        if is_peptide:
            return peptide + train
        else:
            return protein_groups + train
    else:
        if is_peptide:
            return peptide + test
        else:
            return protein_groups + test


def get_save_rootpath(is_peptide: bool, is_cap=False):
    if is_peptide:
        if is_cap:
            return "./count_result/peptide/cap/"
        return "./count_result/peptide/"
    else:
        if is_cap:
            return "./count_result/protein/cap/"
        return "./count_result/protein/"

# def calculate_peptide_overlap(dl: Dict[str, Set], sn: Dict[str, Set]):
    """
        计算某数据集下的肽段层面的各个指标
        
        Retruns:
        ---
        -   dict(str, int): dl, sn 在该数据集下肽段鉴定数量的平均值
        -   dict(str, set): dl, sn 在该数据集下肽段 overlap
        -   dict(str, int): dl, sn 在该数据集下 overlap 的平均值
    """
    # 肽段的统计结果
    # peptide_average, peptide_cup, peptide_cap, peptide_overlap_average = calculate_statistics_attributes(dl, sn)
    # print('#' * 5 + ' peptide_result ' + '#'* 5)
    # print('\t' + 'spectronaut_cup_num: ', len(peptide_cup['spectronaut']), f"spectronaut_average: {peptide_average['spectronaut']}", 'spectronaut_cap_num: ', len(peptide_cap['spectronaut']))
    # print('\t' + 'deeplearning_cup_num: ', len(peptide_cup['deeplearning']), f"deeplearning_average: {peptide_average['deeplearning']}", 'deeplearning_cap_num: ', len(peptide_cap['deeplearning']))
    # print('\t' + 'overlap: ', f'deeplearning_spectronaut_average: {peptide_overlap_average}')
    # print('\t' + 'cap_overlap: 'f'{len(peptide_cap["deeplearning"] & peptide_cap["spectronaut"])}')
    # return peptide_average, peptide_cap, peptide_overlap_average


def peptide_overlap(to_be_overlap: Set[str], to_overlap: Set[str]):
    """
        肽段用于计算 overlap 的函数
    """
    return to_be_overlap & to_overlap


def proteinGroups_overlap(to_be_overlap: Set[str], to_overlap: Set[str]):
    """
        两个 group 中只要有一个蛋白质 map 上就算 overlap

        Parameters:
        ---
        -   to_be_overlap: 待匹配的蛋白质组 set
        -   to_overlap: 需要去匹配的蛋白质组 set

        Returns:
        ---
        -   set: 元素为最后 overlap 的蛋白质组
    """
    group_overlap = set()
    dl_proteins = set()
    # 把所有 deeplearning 的蛋白质筛选出来
    for protein_group in to_be_overlap:
        dl_proteins.update(protein_group.split(';'))

    for protein_group in to_overlap:
        proteins = set(protein_group.split(';'))
        if len(proteins & dl_proteins) != 0:
            group_overlap.add(protein_group)
    return group_overlap


def calculate_peptide_overlap(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]]):
    """
        获得各重复样本的肽段 overlap

        Parameters:
        ---
        -   dl: deeplearning 的结果
        -   sn: spectronaut 的结果

        Returns:
        ---
        -   dict(str, set): key 为文件名, value 为该文件下两种方法鉴定肽段的 overlap
    """
    return calculate_dl_sn_overlap(dl, sn, peptide_overlap)


def calculate_proteinGroups_overlap(dl: Dict[str, Set], sn: Dict[str, Set]):
    """
        获得各重复样本的蛋白质组 overlap

        Parameters:
        ---
        -   dl: deeplearning 的结果
        -   sn: spectronaut 的结果

        Returns:
        ---
        -   dict(str, set): key 为文件名, value 为该文件下两种方法鉴定蛋白质组的 overlap
    """
    groups_overlap = {}
    for file in dl.keys():
        groups_overlap[file] = proteinGroups_overlap(dl[file], sn[file])
    return groups_overlap


def print_metrics(methods_overlap: Dict[str, Set[str]], average: Dict[str, int], final_overlap: Set[str], overlap_average: int):
    """
        输出肽段/蛋白质组的鉴定的统计相关的指标

        1. 各重复样本的平均值, dl/sn
        2. 各重复样本的 overlap 平均值
        3. sn/dl 在所有重复样本的 overlap 
    """
    print("\taverage: ",
          f"deeplearning: {average['dl']}", f"spectronaut: {average['sn']}")
    print("\tmethods_overlap: ", f"deeplearning: {methods_overlap['dl']}",
          f"spectronaut: {methods_overlap['sn']}", f"overlap: {final_overlap}")
    print("\toverlap_average: ", f"{overlap_average}")
    print()


def generate_statics_metrics(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]], overlap_func: Callable[[Set[str], Set[str]], Set[str]]):
    """
        计算肽段/蛋白质组的相关指标

        Returns:
        ---
        dict(): key 值如下所示 
        -   'methods_overlap': 表示两种方法各自在每个样本的 overlap 值
        -   'average': 表示两种方法在重复样本鉴定的平均值
        -   'final_overlap': methods_overlap 中两个方法的结果再取 overlap
        -   'overlap_average': 各重复样本的 overlap 的平均值
    """
    # 两种方法各自计算所有样本的 overlap
    dl_overlap, sn_overlap = calculate_overlap(
        dl, overlap_func), calculate_overlap(sn, overlap_func)
    # 求最后的 overlap
    final_overlap = overlap_func(dl_overlap, sn_overlap)
    # 计算两种方法鉴定的平均值
    dl_average, sn_average = calculate_average(dl), calculate_average(sn)
    # 计算得到两种方法在各重复样本的 overlap 以及平均值
    overlap = calculate_dl_sn_overlap(dl, sn, overlap_func)
    overlap_average = calculate_average(overlap)
    return {
        'methods_overlap': {
            'dl': len(dl_overlap),
            'sn': len(sn_overlap)
        },
        'average': {
            'dl': dl_average,
            'sn': sn_average
        },
        'final_overlap': len(final_overlap),
        'overlap_average': overlap_average
    }


def peptide_statics_metrics(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]]):
    """
        计算肽段相关的指标
    """
    metrics = generate_statics_metrics(dl, sn, peptide_overlap)
    print("#" * 5, " peptide result ", "#" * 5)
    print_metrics(metrics['methods_overlap'], metrics['average'],
                  metrics['final_overlap'], metrics['overlap_average'])
    return metrics


def proteinGroups_statics_metrics(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]]):
    """
        计算蛋白质组相关的指标

        Returns:
        ---
        dict(): key 值如下所示 
        -   'methods_overlap': 表示两种方法各自在每个样本的 overlap 值
        -   'average': 表示两种方法在重复样本鉴定的平均值
        -   'final_overlap': methods_overlap 中两个方法的结果再取 overlap
        -   'overlap_average': overlap 的平均值
    """
    metrics = generate_statics_metrics(dl, sn, proteinGroups_overlap)
    print("#" * 5, " protein_groups result ", "#" * 5)
    print_metrics(metrics['methods_overlap'], metrics['average'],
                  metrics['final_overlap'], metrics['overlap_average'])
    return metrics


def store_metircs(d: Dict[str, List], sn: int, dl: int, overlap: int):
    d['sn'].append(sn)
    d['dl'].append(dl)
    d['overlap'].append(overlap)


def plot_venn(dl_root: str, sn_root: str, datasets: Sequence[str], metrics_func: Callable[[Dict[str, Set[str]], Dict[str, Set[str]]], Dict[str, Any]], is_train: bool, is_peptide: bool):
    files = []  # 存储数据集的名称
    average = {  # 存储两种方法在每个数据集下，重复样本检测数量的平均值
        'sn': [],
        'dl': [],
        'overlap': []
    }
    overlap = {
        'sn': [],
        'dl': [],
        'overlap': []
    }
    for dataset in datasets:
        print('\t' * 2 + '#' * 5 +
              f" {dataset.split('.')[0]}_result " + '#' * 5)
        dl, sn = load_dict_data(
            dl_root + dataset), load_dict_data(sn_root + dataset)
        metrics = metrics_func(dl, sn)
        files.append(dataset.split('.')[0])
        # 均值, 以及 overlap 的均值
        store_metircs(average, metrics['average']['sn'],
                      metrics['average']['dl'], metrics['overlap_average'])
        ###
        store_metircs(overlap, metrics['methods_overlap']['sn'],
                      metrics['methods_overlap']['dl'], metrics['final_overlap'])

    # 平均值图
    fig_title, save_path = get_fig_title(
        is_train, is_peptide), get_save_rootpath(is_peptide)
    plot_venn_figure(average['dl'], average['sn'],
                     average['overlap'], files, is_train, fig_title, save_path)
    # overlap 图
    fig_title, save_path = get_fig_title(
        is_train, is_peptide), get_save_rootpath(is_peptide, True)
    plot_venn_figure(overlap['dl'], overlap['sn'],
                     overlap['overlap'], files, is_train, fig_title, save_path)


def plot_proteinGroups_venn(datasets: Sequence[str], is_train: bool):
    dl_root = "./protein_result/deeplearning/"
    sn_root = "./protein_result/spectronaut/"
    plot_venn(dl_root, sn_root, datasets,
              proteinGroups_statics_metrics, is_train, False)


def plot_peptide_venn(datasets: Sequence[str], is_train: bool):
    dl_root = "./peptide_result/deeplearning/"
    sn_root = "./peptide_result/spectronaut/"
    plot_venn(dl_root, sn_root, datasets,
              peptide_statics_metrics, is_train, True)


def generate_result(is_train: bool = True):
    if is_train:
        print('\t' * 5 + '#' * 5 + " train_Data " + '#' * 5)
    else:
        print('\t' * 5 + '#' * 5 + " test_Data " + '#' * 5)

    s, e = "dataset", ".npy"
    train_dataset_n, test_dataset_n = list(range(1, 7)), list(range(7, 10))
    train_dataset = [s + str(i) + e for i in train_dataset_n]
    test_dataset = [s + str(i) + e for i in test_dataset_n]
    if is_train:
        plot_peptide_venn(train_dataset, is_train)
        plot_proteinGroups_venn(train_dataset, is_train)
    else:
        plot_peptide_venn(test_dataset, is_train)
        plot_proteinGroups_venn(test_dataset, is_train)


if __name__ == "__main__":
    generate_result(True)
    generate_result(False)
