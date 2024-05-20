from typing import Dict, Set, List, Callable, Sequence, Any, Literal

import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from ..generatedata.utils.utils import read_dict_npy


def add_item_to_dict(dict_name: Dict[str, Set[str]], key: str, item: str):
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
    proteins = protein_group.split(';')
    proteins.sort()
    sorted_protein_group = ';'.join(proteins)
    return sorted_protein_group


def generate_sepctronaut_peptideOrProteinGroups_file_set(df: pd.DataFrame, file_names: List[str], column: str):
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
    peptide_file_set = set()
    for file_name, _, modified_peptide_charge in fdr_filter_target:
        stripped_peptide = modifiedPeptide_to_strippedPeptide_map[modified_peptide_charge[0]]
        peptide_file_set.add((stripped_peptide, file_name))
    return peptide_file_set


def filter_data_by_threshold(d: Dict[str, Set[str]], threshold: int):
    wipe_d: Dict[str, Set[str]] = {}
    for key, val in d.items():
        if len(val) < threshold:
            continue
        wipe_d[key] = val
    return wipe_d


def generate_file_valueSet_dict(peptideOrProteinGroups_file_set: Set):
    file_valueSet_dict: Dict[str, Set[str]] = {
        file_name: set() for _, file_name in peptideOrProteinGroups_file_set}
    for value, file_name in peptideOrProteinGroups_file_set:
        add_item_to_dict(file_valueSet_dict, file_name, value)
    return file_valueSet_dict


def generate_modifiedPeptide_to_strippedPeptide_map(library: pd.DataFrame):
    modifiedPeptide_to_strippedPeptide_map = {}
    for modified_peptide, stripped_peptide in library[['ModifiedPeptide', 'StrippedPeptide']].values:
        modifiedPeptide_to_strippedPeptide_map[modified_peptide] = stripped_peptide
    return modifiedPeptide_to_strippedPeptide_map


def read_deeplearning(target, decoy, modifiedPeptide_to_strippedPeptide_map: Dict):
    threshold, fdr_target = fdr(target, decoy)
    peptide_file_set = generate_deeplearning_peptide_file_set(
        fdr_target, modifiedPeptide_to_strippedPeptide_map)
    filename_to_valueset = generate_file_valueSet_dict(peptide_file_set)
    return threshold, filter_data_by_threshold(filename_to_valueset, 200)


def calculate_cup(d: Dict[str, Set[str]]):
    files = list(d.keys())
    initial_file = files[0]
    cup = d[initial_file].copy()
    for f in files[1:]:
        cup |= d[f]
    return cup


def calculate_overlap(d: Dict[str, Set[str]], overlap_func: Callable[[Set[str], Set[str]], Set[str]]):
    files = list(d.keys())
    initial_file = files[0]
    cap = d[initial_file].copy()
    for _, f in enumerate(files[1:]):
        cap = overlap_func(cap, d[f])
    return cap


def calculate_dl_sn_overlap(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]], overlap_func: Callable[[Set[str], Set[str]], Set[str]]):
    overlap: Dict[str, Set[str]] = {}
    for file in dl.keys():
        overlap[file] = overlap_func(dl[file], sn[file])
    return overlap


def calculate_average(d: Dict[str, Set[str]]):
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
    return to_be_overlap & to_overlap


def proteinGroups_overlap(to_be_overlap: Set[str], to_overlap: Set[str]):
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
    return calculate_dl_sn_overlap(dl, sn, peptide_overlap)


def calculate_proteinGroups_overlap(dl: Dict[str, Set], sn: Dict[str, Set]):
    groups_overlap = {}
    for file in dl.keys():
        groups_overlap[file] = proteinGroups_overlap(dl[file], sn[file])
    return groups_overlap


def print_metrics(methods_overlap: Dict[str, Set[str]], average: Dict[str, int], final_overlap: Set[str], overlap_average: int):
    print("\taverage: ",
          f"deeplearning: {average['dl']}", f"spectronaut: {average['sn']}")
    print("\tmethods_overlap: ", f"deeplearning: {methods_overlap['dl']}",
          f"spectronaut: {methods_overlap['sn']}", f"overlap: {final_overlap}")
    print("\toverlap_average: ", f"{overlap_average}")
    print()


def generate_statics_metrics(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]], overlap_func: Callable[[Set[str], Set[str]], Set[str]]):
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
    metrics = generate_statics_metrics(dl, sn, peptide_overlap)
    print("#" * 5, " peptide result ", "#" * 5)
    print_metrics(metrics['methods_overlap'], metrics['average'],
                  metrics['final_overlap'], metrics['overlap_average'])
    return metrics


def proteinGroups_statics_metrics(dl: Dict[str, Set[str]], sn: Dict[str, Set[str]]):
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
        dest[v[0]][type]['info'].append((v[1][0], v[1][1], v[2]))


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
