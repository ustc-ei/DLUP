from typing import Dict, Set, List, Callable, Sequence, Any, Literal

import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from ..generatedata.utils.utils import read_dict_npy


def add_item_to_dict(dict_name: Dict[str, Set[str]], key: str, item: str):
<<<<<<< HEAD
    """
        将 item 添加到字典中对应 key 的 set 中
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    dict_name[key].add(item)


def fdr(target, decoy):
    target_score, target_info = target['score'], target['info']
    decoy_score, _ = decoy['score'], decoy['info']

    t, d = np.sort(target_score), np.sort(decoy_score)

    threshold = t[0]

    # 计算界限值
    d_i = 0
    for i in tqdm(range(len(t))):
        while d[d_i] < t[i]:
            d_i += 1
        fdr = (len(d) - d_i) / (len(t) - i)
        if fdr < 0.01:
            threshold = t[i]
            break

    return threshold, target_info[target_score >= threshold]


def sort_protein_in_group(protein_group: str):
<<<<<<< HEAD
    """
        将蛋白质组中的蛋白质进行排序, 方便后面筛选

    Parameters:
    ---
    -   protein_group: 蛋白质组

    Returns:
    ---
    -   str: 排序之后的蛋白质组
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    proteins = protein_group.split(';')
    proteins.sort()
    sorted_protein_group = ';'.join(proteins)
    return sorted_protein_group


def generate_sepctronaut_peptideOrProteinGroups_file_set(df: pd.DataFrame, file_names: List[str], column: str):
<<<<<<< HEAD
    """
        生成 spectronaut 肽段/蛋白质及其对应文件, 集合中的元素为 (peptide/protein, file_name)

        Paramterts:
        ---
        -   df: spectronaut 对应的 dataframe
        -   filenames: spectronaut 测定的文件序列
        -   column: 需要提取的元素对应的列名, 肽段/蛋白质, 'PEP.StrippedSequence'/'PG.ProteinGroups'

        Returns:
        ---
        -   set(): 元素为 (peptide/protein, file_name) 的集合
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    peptideOrProteinGroups_file_set = set()
    # 前面的 column_name
    # ['PG.ProteinGroups', 'PG.ProteinAccessions', 'PG.Genes', 'PG.ProteinDescriptions', 'PEP.StrippedSequence', 'EG.PrecursorId']
    for i, row_data in enumerate(df.values):
        column_data = df.iloc[i][column]  # type: ignore
        for j, file_data in enumerate(row_data[6:]):
            if not np.isnan(file_data):  # 把其中非 nan 的值筛选出来
                file_name = file_names[j].split('.')[0].split(']')[1].strip()
                peptideOrProteinGroups_file_set.add(
                    (sort_protein_in_group(column_data), file_name))
    return peptideOrProteinGroups_file_set


def read_spectronaut(file_path: str, library_strippedsequence: Set[str], is_protein: bool = False):
<<<<<<< HEAD
    """
        读取 spectronaut 的结果

        Parameters:
        ---
        -   file_path: spectronaut 结果路径
        -   is_protein: 是否是蛋白质结果
        Returns:
        ---
        -   dict(str, set): key 为文件名, value 为结果的 set 
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    df = pd.read_csv(file_path, sep='\t')
    file_names = list(df.columns[6:])
    column = "PEP.StrippedSequence"
    if is_protein:
        column = "PG.ProteinGroups"
    column_file_set = generate_sepctronaut_peptideOrProteinGroups_file_set(
        df, file_names, column)
    filename_to_valueset = generate_file_valueSet_dict(column_file_set)
    for k, val in filename_to_valueset.items():
        filename_to_valueset[k] = set(
            [lib for lib in val if lib in library_strippedsequence])
    return filter_data_by_threshold(filename_to_valueset, 200)


def generate_deeplearning_peptide_file_set(fdr_filter_target: npt.NDArray, modifiedPeptide_to_strippedPeptide_map: Dict[str, str]):
<<<<<<< HEAD
    """
        生成 deeplearning 肽段及其对应文件, 集合中的元素为 (peptide, file_name)

        Paramterts:
        ---
        -   fdr_filter_target: 经过质量控制之后的结果
        -   modifiedPeptide_to_strippedPeptide_map: modified_peptide -> stripped_peptide 的字典映射 

        Returns:
        ---
        -   set(): 元素为 (peptide, file_name) 的集合
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    peptide_file_set = set()
    for file_name, _, modified_peptide_charge in fdr_filter_target:
        stripped_peptide = modifiedPeptide_to_strippedPeptide_map[modified_peptide_charge[0]]
        peptide_file_set.add((stripped_peptide, file_name))
    return peptide_file_set


def filter_data_by_threshold(d: Dict[str, Set[str]], threshold: int):
<<<<<<< HEAD
    """
        删除某些数据差别过大的结果

        Parameters:
        ---
        -   d: 初始数据 dict(): key 为文件名, value 为对应的元素 肽段/蛋白质

        Returns:
        --- 
        -   dict(): 删除过后的数据, key 数量减少或不变
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    wipe_d: Dict[str, Set[str]] = {}
    for key, val in d.items():
        if len(val) < threshold:
            continue
        wipe_d[key] = val
    return wipe_d


def generate_file_valueSet_dict(peptideOrProteinGroups_file_set: Set):
<<<<<<< HEAD
    """
        将之前生成 (peptide/Protein, file_name) 的 set 转化为 key 为 file_name, value 为 set 的字典

        Parameters:
        ---
        -   peptideOrProteinGroups_file_set: 需要转换的 set

        Returns:
        ---
        -   dict(): key 为 file_name, value 为 set
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    file_valueSet_dict: Dict[str, Set[str]] = {
        file_name: set() for _, file_name in peptideOrProteinGroups_file_set}
    for value, file_name in peptideOrProteinGroups_file_set:
        add_item_to_dict(file_valueSet_dict, file_name, value)
    return file_valueSet_dict


def generate_modifiedPeptide_to_strippedPeptide_map(library: pd.DataFrame):
<<<<<<< HEAD
    """
        生成图谱库中的修饰肽段 -> 肽段序列的字典

        由于 spectronaut 中结果可能出现修饰肽段不在图谱库的情况, 因此需要都映射成 StrippedPeptide

        spectronaut 结果本身就有 StrippedPeptide，因此暂且不用管, 需要把 deeplearning 的结果中的 modifiedPeptide 映射为 StrippedPeptide

        Parameters:
        ---
        -   library: 读取的图谱库

        Returns:
        ---
        -   dict(): key 为 ModifiedPeptide, value 为 StrippedPeptide  
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    modifiedPeptide_to_strippedPeptide_map = {}
    for modified_peptide, stripped_peptide in library[['ModifiedPeptide', 'StrippedPeptide']].values:
        modifiedPeptide_to_strippedPeptide_map[modified_peptide] = stripped_peptide
    return modifiedPeptide_to_strippedPeptide_map


def read_deeplearning(target, decoy, modifiedPeptide_to_strippedPeptide_map: Dict):
<<<<<<< HEAD
    """
        读取 deeplearning 的结果

        Parameters:
        ---
        -   target_path: 目标库结果
        -   decoy_path: 诱饵库结果
        -   modifiedPeptide_to_strippedPeptide_map: modified_peptide -> stripped_peptide 的映射字典

        Returns:
        -   dict(str, set): key 为文件名, value 为对应文件结果的 set
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    threshold, fdr_target = fdr(target, decoy)
    peptide_file_set = generate_deeplearning_peptide_file_set(
        fdr_target, modifiedPeptide_to_strippedPeptide_map)
    filename_to_valueset = generate_file_valueSet_dict(peptide_file_set)
    return threshold, filter_data_by_threshold(filename_to_valueset, 200)


def calculate_cup(d: Dict[str, Set[str]]):
<<<<<<< HEAD
    """
        计算各重复样本结果的并集
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    files = list(d.keys())
    initial_file = files[0]
    cup = d[initial_file].copy()
    for f in files[1:]:
        cup |= d[f]
    return cup


def calculate_overlap(d: Dict[str, Set[str]], overlap_func: Callable[[Set[str], Set[str]], Set[str]]):
<<<<<<< HEAD
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
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    files = list(d.keys())
    initial_file = files[0]
    cap = d[initial_file].copy()
    for _, f in enumerate(files[1:]):
        cap = overlap_func(cap, d[f])
    return cap


def calculate_dl_sn_overlap(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]], overlap_func: Callable[[Set[str], Set[str]], Set[str]]):
<<<<<<< HEAD
    """
        计算 dl, sn 两种方法在各重复样本下的 overlap
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    overlap: Dict[str, Set[str]] = {}
    for file in dl.keys():
        overlap[file] = overlap_func(dl[file], sn[file])
    return overlap


def calculate_average(d: Dict[str, Set[str]]):
<<<<<<< HEAD
    """
        计算各重复样本下的鉴定数量的平均值
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
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


def peptide_overlap(to_be_overlap: Set[str], to_overlap: Set[str]):
<<<<<<< HEAD
    """
        肽段用于计算 overlap 的函数
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    return to_be_overlap & to_overlap


def proteinGroups_overlap(to_be_overlap: Set[str], to_overlap: Set[str]):
<<<<<<< HEAD
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
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
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
<<<<<<< HEAD
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
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    return calculate_dl_sn_overlap(dl, sn, peptide_overlap)


def calculate_proteinGroups_overlap(dl: Dict[str, Set], sn: Dict[str, Set]):
<<<<<<< HEAD
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
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    groups_overlap = {}
    for file in dl.keys():
        groups_overlap[file] = proteinGroups_overlap(dl[file], sn[file])
    return groups_overlap


def print_metrics(methods_overlap: Dict[str, Set[str]], average: Dict[str, int], final_overlap: Set[str], overlap_average: int):
<<<<<<< HEAD
    """
        输出肽段/蛋白质组的鉴定的统计相关的指标

        1. 各重复样本的平均值, dl/sn
        2. 各重复样本的 overlap 平均值
        3. sn/dl 在所有重复样本的 overlap 
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    print("\taverage: ",
          f"deeplearning: {average['dl']}", f"spectronaut: {average['sn']}")
    print("\tmethods_overlap: ", f"deeplearning: {methods_overlap['dl']}",
          f"spectronaut: {methods_overlap['sn']}", f"overlap: {final_overlap}")
    print("\toverlap_average: ", f"{overlap_average}")
    print()


def generate_statics_metrics(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]], overlap_func: Callable[[Set[str], Set[str]], Set[str]]):
<<<<<<< HEAD
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
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
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
<<<<<<< HEAD
    """
        计算肽段相关的指标
    """
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    metrics = generate_statics_metrics(dl, sn, peptide_overlap)
    print("#" * 5, " peptide result ", "#" * 5)
    print_metrics(metrics['methods_overlap'], metrics['average'],
                  metrics['final_overlap'], metrics['overlap_average'])
    return metrics


def proteinGroups_statics_metrics(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]]):
<<<<<<< HEAD
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
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    metrics = generate_statics_metrics(dl, sn, proteinGroups_overlap)
    print("#" * 5, " protein_groups result ", "#" * 5)
    print_metrics(metrics['methods_overlap'], metrics['average'],
                  metrics['final_overlap'], metrics['overlap_average'])
    return metrics


def store_metircs(d: Dict[str, List], sn: int, dl: int, overlap: int):
    d['sn'].append(sn)
    d['dl'].append(dl)
    d['overlap'].append(overlap)


def split_score_by_files(dest, src, type: Literal['target', 'decoy']):
    for i, v in enumerate(src['info']):
        # v[0] 表示的是 file
        dest[v[0]][type]['score'].append(src['score'][i])
<<<<<<< HEAD
        dest[v[0]][type]['info'].append((v[1][0], v[1][1]))
=======
        dest[v[0]][type]['info'].append((v[1][0], v[1][1], v[2]))
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79


def get_dl_identify(target_path: str, decoy_path: str, modifiedPeptide_to_strippedPeptide_map: Dict[str, str]):
    t = read_dict_npy(target_path)
    d = read_dict_npy(decoy_path)
    predicted_info = {
        v[0]: {
            'target': {
                'score': [],
                'info': []
            },
            'decoy': {
                'score': [],
                'info': []
            }
        }
        for v in t['info']
    }
    split_score_by_files(predicted_info, d, 'decoy')
    split_score_by_files(predicted_info, t, 'target')
    for v in predicted_info.values():
        v['target']['score'] = np.array(v['target']['score'])  # type: ignore
        v['decoy']['score'] = np.array(v['decoy']['score'])  # type: ignore
        v['target']['info'] = np.array(  # type: ignore
            v['target']['info'], dtype=object)  # type: ignore
        v['decoy']['info'] = np.array(  # type: ignore
            v['decoy']['info'], dtype=object)  # type: ignore
    r = {}
    for k, v in predicted_info.items():
        print(k)
        threshold, result = fdr(v['target'], v['decoy'])
        print(f"threshold={threshold}, modified_peptide_length={len(result)}")
        r[k] = set()
        for item in result:
            r[k].add(modifiedPeptide_to_strippedPeptide_map[item[2][0]])
    return r
<<<<<<< HEAD

def get_dl_identify_modified(t: npt.NDArray, d: npt.NDArray):
    predicted_info = {
        v[0]: {
            'target': {
                'score': [],
                'info': []
            },
            'decoy': {
                'score': [],
                'info': []
            }
        }
        for v in t['info']
    }
    split_score_by_files(predicted_info, d, 'decoy')
    split_score_by_files(predicted_info, t, 'target')
    for v in predicted_info.values():
        v['target']['score'] = np.array(v['target']['score'])
        v['decoy']['score'] = np.array(v['decoy']['score'])
        v['target']['info'] = np.array(v['target']['info'], dtype=object)
        v['decoy']['info'] = np.array(v['decoy']['info'], dtype=object)
    r = {}
    for k, v in predicted_info.items():
        print(k)
        threshold, result = fdr(v['target'], v['decoy'])
        print(f"threshold={threshold}, modified_peptide_length={len(result)}")
        r[k] = set()
        for item in result:
            r[k].add(tuple(item))
    return r
=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
