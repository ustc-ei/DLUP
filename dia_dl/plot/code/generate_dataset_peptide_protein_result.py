import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing import List, Set, Dict
import idpicker


def add_item_to_dict(dict_name: Dict[str, Set[str]], key: str, item: str):
    """
        将 item 添加到字典中对应 key 的 set 中
    """
    dict_name[key].add(item)


def generate_modifiedPeptide_to_strippedPeptide_map(library: pd.DataFrame):
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
    modifiedPeptide_to_strippedPeptide_map = {}
    for modified_peptide, stripped_peptide in library[['ModifiedPeptide', 'StrippedPeptide']].values:
        modifiedPeptide_to_strippedPeptide_map[modified_peptide] = stripped_peptide
    return modifiedPeptide_to_strippedPeptide_map


def filter_data_by_threshold(d: Dict[str, Set[str]], threshold: int):
    """
        删除某些数据差别过大的结果

        Parameters:
        ---
        -   d: 初始数据 dict(): key 为文件名, value 为对应的元素 肽段/蛋白质

        Returns:
        --- 
        -   dict(): 删除过后的数据, key 数量减少或不变
    """
    wipe_d: Dict[str, Set[str]] = {}
    for key, val in d.items():
        if len(val) < threshold:
            continue
        wipe_d[key] = val
    return wipe_d


def generate_file_valueSet_dict(peptideOrProteinGroups_file_set: Set):
    """
        将之前生成 (peptide/Protein, file_name) 的 set 转化为 key 为 file_name, value 为 set 的字典

        Parameters:
        ---
        -   peptideOrProteinGroups_file_set: 需要转换的 set

        Returns:
        ---
        -   dict(): key 为 file_name, value 为 set
    """
    file_valueSet_dict: Dict[str, Set[str]] = {
        file_name: set() for _, file_name in peptideOrProteinGroups_file_set}
    for value, file_name in peptideOrProteinGroups_file_set:
        add_item_to_dict(file_valueSet_dict, file_name, value)
    return file_valueSet_dict


def generate_sepctronaut_peptideOrProteinGroups_file_set(df: pd.DataFrame, file_names: List[str], column: str):
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
    peptideOrProteinGroups_file_set = set()
    # 前面的 column_name
    # ['PG.ProteinGroups', 'PG.ProteinAccessions', 'PG.Genes', 'PG.ProteinDescriptions', 'PEP.StrippedSequence', 'EG.PrecursorId']
    for i, row_data in enumerate(df.values):
        column_data = df.iloc[i][column]  # type: ignore
        for j, file_data in enumerate(row_data[6:]):
            if not np.isnan(file_data):  # 把其中非 nan 的值筛选出来
                file_name = file_names[j].split('.d')[0].split(']')[1].strip()
                peptideOrProteinGroups_file_set.add(
                    (sort_protein_in_group(column_data), file_name))
    return peptideOrProteinGroups_file_set


def read_spectronaut(file_path: str, is_protein: bool = False):
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
    df = pd.read_csv(file_path, sep='\t')
    file_names = list(df.columns[6:])
    column = "PEP.StrippedSequence"
    if is_protein:
        column = "PG.ProteinGroups"
    column_file_set = generate_sepctronaut_peptideOrProteinGroups_file_set(
        df, file_names, column)
    filename_to_valueset = generate_file_valueSet_dict(column_file_set)
    return filter_data_by_threshold(filename_to_valueset, 200)


def fdr_filter(target_path: str, decoy_path: str):
    """
        FDR 质量控制操作, 筛选出大于分数阈值的目标肽段及其对应文件

        Parameters:
        ---
        -   target_path: 目标库结果路径
        -   decoy_path: 诱饵库结果路径

        Returns:
        ---
        -   FDR 质量控制过后的肽段结果及对应的文件名
    """
    target_file, target = np.load(target_path, allow_pickle=True)
    _, decoy = np.load(decoy_path, allow_pickle=True)

    target_score = []
    decoy_score = []

    for score in target:
        target_score.extend(score)

    for score in decoy:
        decoy_score.extend(score)

    target_score = np.array(target_score)
    decoy_score = np.array(decoy_score)

    target_sort = np.sort(target_score)
    decoy_sort = np.sort(decoy_score)

    threshold = target_sort[0]

    # 计算界限值
    d_i = 0
    for i in tqdm(range(len(target_sort))):
        while decoy_sort[d_i] < target_sort[i]:
            d_i += 1
        fdr = (len(decoy_sort) - d_i) / (len(target_sort) - i)
        if fdr < 0.01:
            threshold = target_sort[i]
            break

    # 筛选出大于阈值的肽段及其所在文件
    return target_file[(target_score >= threshold).nonzero()]


def generate_deeplearning_peptide_file_set(fdr_filter_target: npt.NDArray, modifiedPeptide_to_strippedPeptide_map: Dict[str, str]):
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
    peptide_file_set = set()
    for file_name, modified_peptide_charge in fdr_filter_target:
        stripped_peptide = modifiedPeptide_to_strippedPeptide_map[modified_peptide_charge[0]]
        peptide_file_set.add((stripped_peptide, file_name))
    return peptide_file_set


def read_deeplearning(target_path: str, decoy_path: str, modifiedPeptide_to_strippedPeptide_map: Dict):
    """
        读取 deeplearning 的结果

        Parameters:
        ---
        -   target_path: 目标库结果路径
        -   decoy_path: 诱饵库结果路径
        -   modifiedPeptide_to_strippedPeptide_map: modified_peptide -> stripped_peptide 的映射字典

        Returns:
        -   dict(str, set): key 为文件名, value 为对应文件结果的 set
    """
    fdr_filter_target = fdr_filter(target_path, decoy_path)
    peptide_file_set = generate_deeplearning_peptide_file_set(
        fdr_filter_target, modifiedPeptide_to_strippedPeptide_map)
    filename_to_valueset = generate_file_valueSet_dict(peptide_file_set)
    return filter_data_by_threshold(filename_to_valueset, 200)


def generate_strippedPeptide_to_proteinGroups_map(library: pd.DataFrame):
    """
        根据图谱库生成 stripped_peptide -> proteinGroups 的映射字典

        Parameters:
        ---
        -   library: 读取的图谱库

        Returns:
        -   dict(str, str): key 为 stripped_peptide, value 为其对应的 protein_groups
    """
    strippedPeptide_to_proteinGroups_map = {}
    for stripped_peptide, protein_groups in library[['StrippedPeptide', 'ProteinGroups']].values:
        if stripped_peptide not in strippedPeptide_to_proteinGroups_map.keys():
            strippedPeptide_to_proteinGroups_map[stripped_peptide] = protein_groups
    return strippedPeptide_to_proteinGroups_map


def opt_data_for_files_overlap(previous_dict: Dict[str, Set[str]], files_overlap: Set[str]):
    """
        筛选出共有文件的数据
    """
    new_dict = {}
    for file in files_overlap:
        new_dict[file] = previous_dict[file]
    return new_dict


def calculate_files_overlap(deeplearning: Dict, spectronaut: Dict) -> Set[str]:
    """
        获得两者共有的文件名

        Parameters:
        ---
        -   deeplearning: deeplearning 对应结果
        -   spectronaut: spectronaut 对应结果

        Returns:
        ---
        -   set: 共有的文件名
    """
    deeplearning_files = set([file for file in deeplearning.keys()])
    spectronaut_files = set([file for file in spectronaut.keys()])
    files = deeplearning_files & spectronaut_files
    return files


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


def normlize_proteinGroups(protein_groups: Set[str]):
    """
        规范化蛋白质组的标识符

        idpicker 输出的蛋白质组的结果为 num/protein_1/protein_2/.../protein_num, num 为该蛋白质组中蛋白质的数量, 后面紧接着的是蛋白质

        我们需要把上面这种形式变为  protein 或 protein_1;protein_2;...;protein_num, 和 spectronaut 的结果保持一致

        Parameters:
        ---
        -   protein_groups: idpicker 算法输出的蛋白质组结果

        Returns:
        ---
        -   set: 规划化过后的蛋白质组结果
    """
    normlized_protein_groups: Set[str] = set()
    for protein_group in protein_groups:
        normlized_protein_group = ";".join(protein_group.split('/')[1:])
        normlized_protein_groups.add(
            sort_protein_in_group(normlized_protein_group))
    return normlized_protein_groups


def find_proteins(peptide_set: Set[str], strippedPeptide_to_proteinGroups_map: Dict[str, str]):
    """
        使用 idpicker 算法得到蛋白质组的结果

        Parameters:
        ---
        -   peptide_set: 肽段的结果
        -   strippedPeptide_to_proteinGroups_map: stripped_peptide -> protein_groups 的映射字典

        Returns:
        ---
        -  set: 名字规范化后的蛋白质组
    """
    peptide_protein_edge_list = []
    for peptide in peptide_set:
        protein_groups = strippedPeptide_to_proteinGroups_map[peptide].split(
            ';')
        for protein in protein_groups:
            peptide_protein_edge_list.append((peptide, protein))
    # 去重, 防止添加重复的边
    protein_groups = set(idpicker.find_valid_proteins(
        list(set(peptide_protein_edge_list))).values())
    # 名字规范化
    protein_groups = normlize_proteinGroups(protein_groups)
    return protein_groups


def is_have_same_protein_in_proteinGroups(r: Dict):
    """
        判断是否存在两个不同的蛋白质组中存在两个相同的蛋白质
    """
    for value in r.values():
        d = {}
        for protein_groups in value:
            for protein in protein_groups.split(';'):
                if protein not in d.keys():
                    d[protein] = protein_groups
                elif protein_groups != d[protein]:
                    print("two protein_groups have the same protein!")


def save_peptide_proteinGroups(dl_peptide: Dict[str, Set[str]], sn_peptide: Dict[str, Set[str]], sn_protein: Dict[str, Set[str]], peptide_protein_map: Dict[str, str], dataset_name: str):
    """
        保存 deeplearning 和 spectronaut 的肽段及蛋白质组的结果

        Parameters:
        ---
        -   dl_peptide: deeplearning 的肽段结果, key 为 file_name, value 为 stripped_peptide 的 set
        -   sn_peptide: spectronaut 的肽段结果, 同上
        -   sn_protein: spectronaut 的蛋白质组结果, key 为 file_name, value 为 protein_groups 的 set
        -   peptide_protein_map: stripped_peptide -> protein_groups 的映射字典
        -   dataset_name: dataset 的名字
    """

    # 筛选出公有的文件
    files_overlap = calculate_files_overlap(dl_peptide, sn_peptide)
    print(len(files_overlap))
    # 筛选肽段以及蛋白质组的结果
    sn_peptide = opt_data_for_files_overlap(sn_peptide, files_overlap)
    sn_protein = opt_data_for_files_overlap(sn_protein, files_overlap)
    is_have_same_protein_in_proteinGroups(sn_protein)
    dl_peptide = opt_data_for_files_overlap(dl_peptide, files_overlap)
    dl_protein = {
        file: find_proteins(peptide_set, peptide_protein_map)
        for file, peptide_set in dl_peptide.items()
    }

    np.save(f"./peptide_result/spectronaut/{dataset_name}.npy", sn_peptide)
    np.save(f"./peptide_result/deeplearning/{dataset_name}.npy", dl_peptide)
    np.save(f"./protein_result/deeplearning/{dataset_name}.npy", dl_protein)
    np.save(f"./protein_result/spectronaut/{dataset_name}.npy", sn_protein)


def generate_result(is_train: bool = True):
    """
        生成两种方法的肽段及蛋白质组的结果

        Parameters:
        ---
        -   is_train: 是否是训练集
    """
    root = './train_result/'
    if not is_train:
        root = './test_result/'
    if is_train:
        print('\t' * 5 + '#' * 10 + ' TrainDataResult ' + '#' * 10)
    else:
        print('\t' * 5 + '#' * 10 + ' TestDataResult ' + '#' * 10)

    library = pd.read_csv('./library/AD8-300S-directDIA.tsv', sep='\t')
    modifiedPeptide_to_strippedPeptide_map = generate_modifiedPeptide_to_strippedPeptide_map(
        library)
    strippedPeptide_to_proteinGroups_map = generate_strippedPeptide_to_proteinGroups_map(
        library)

    for dataset_dir in os.listdir(root):
        dataset_name = dataset_dir.split('_')[0]

        print('#' * 5 + f" {dataset_name}_result " + '#' * 5)

        dataset_dir = root + dataset_dir + '/'

        sn_filePath = [
            dataset_dir + file for file in os.listdir(dataset_dir) if file.endswith('.tsv')][0]

        dl_target_filePath = [
            dataset_dir + file for file in os.listdir(dataset_dir) if 'target' in file][0]
        dl_decoy_filePath = [
            dataset_dir + file for file in os.listdir(dataset_dir) if 'decoy' in file][0]
        # 读取 spectronaut 的肽段及蛋白质结果
        sn_peptide = read_spectronaut(sn_filePath)
        sn_protein = read_spectronaut(sn_filePath, True)
        # 读取 deeplearning 的肽段结果
        dl_peptide = read_deeplearning(
            dl_target_filePath, dl_decoy_filePath, modifiedPeptide_to_strippedPeptide_map)
        # 保存两者的肽段、蛋白质组结果
        save_peptide_proteinGroups(
            dl_peptide, sn_peptide, sn_protein, strippedPeptide_to_proteinGroups_map, dataset_name)


if __name__ == "__main__":
    generate_result(True)
    generate_result(False)
