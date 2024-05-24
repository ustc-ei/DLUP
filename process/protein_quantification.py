from typing import Set, Dict, Sequence
from functools import reduce

import numpy as np
import pandas as pd

from .idpicker import find_valid_proteins


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
    protein_groups = set(
        find_valid_proteins(list(set(peptide_protein_edge_list))).values()
    )
    # 名字规范化
    protein_groups = normlize_proteinGroups(protein_groups)
    return protein_groups


def generate_peptide_protein_bidict(
    target_library: Dict,
):
    strippedsequence_to_proteingroups = {}
    proteingroups_to_modifiedcharge = {
        target_library[key]['ProteinGroups']: []
        for key in target_library.keys()
    }
    for modified_charge, metadata in target_library.items():
        stripped_sequence = metadata['StrippedPeptide']
        protein_groups = metadata['ProteinGroups']
        strippedsequence_to_proteingroups[stripped_sequence] = protein_groups
        proteingroups_to_modifiedcharge[protein_groups].append(modified_charge)
    return strippedsequence_to_proteingroups, proteingroups_to_modifiedcharge


def merge(df1, df2, on_columns: Sequence[str]):
    df = pd.merge(df1, df2, how='outer', on=on_columns)
    return df


def quantification(
    peptide_quant: Dict,
    library_path: str,
    save_path: str
):
    proteingroups_identification = {}
    proteingroups_quantification = {}

    strippedsequence_to_proteingroups, proteingroups_to_modifiedcharge = generate_peptide_protein_bidict(
        library_path)

    for file, value in peptide_quant.items():
        # 得到各个文件的蛋白质组的定性结果
        stripped_sequence_set = set()

        for _, info in value.items():
            stripped_sequence = info['strippedSequence']
            stripped_sequence_set.add(stripped_sequence)

        proteingroups_identification[file] = find_proteins(
            stripped_sequence_set, strippedsequence_to_proteingroups)
        proteingroups_quantification[file] = {}

        for protein_group in proteingroups_identification[file]:
            protein_quantity = np.sum(
                [value[p]['quantity'] for p in proteingroups_to_modifiedcharge[protein_group] if p in value])
            proteingroups_quantification[file][protein_group] = protein_quantity

        df_sequence = []

        for file, quant in proteingroups_quantification.items():
            quantity_col = f'{file}.Quantity'
            df = pd.DataFrame(columns=['ProteinGroups', quantity_col])
            for protein_group, quantity in quant.items():
                # df 插入一条数据, 需要使用定性的结果进行筛选
                # 1. 筛选出定性到的肽段
                # 2. 将同一个的 stripped_sequence 肽段的定量结果进行合并
                df.loc[len(df)] = [
                    protein_group,
                    quantity
                ]

            df_sequence.append(df)

    df_generator = (df for df in df_sequence)
    df = reduce(lambda x, y: merge(x, y, ['ProteinGroups']), df_generator)
    df.to_csv(save_path, index=False, sep='\t')
