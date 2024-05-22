from typing import Dict, Sequence
from functools import reduce

import pandas as pd

def merge(df1, df2, on_columns: Sequence[str]):
    df = pd.merge(df1, df2, how='outer', on=on_columns)
    return df

def quantification(
    quant: Dict,
    save_path: str
):
    df_sequence = []
    for file, pep_quant_dict in quant.items():
        quantity_col = f'{file}.Quantity'
        df = pd.DataFrame(columns=['modifiedPeptide', 'charge', 'strippedSequence', quantity_col])
        for modified_charge, info in pep_quant_dict.items():
            # df 插入一条数据, 需要使用定性的结果进行筛选
            # 1. 筛选出定性到的肽段
            # 2. 将同一个的 stripped_sequence 肽段的定量结果进行合并
            df.loc[len(df)] = [
                modified_charge[0],
                modified_charge[1],
                info['strippedSequence'],
                info['quantity']
            ]
        groups = df.groupby('strippedSequence')

        for k, v in groups.groups.items():
            group = groups.get_group(k)
            df.loc[v, quantity_col] = group[quantity_col].sum()
        df_sequence.append(df)

    on_columns = ['modifiedPeptide', 'charge', 'strippedSequence']
    df_generator = (df for df in df_sequence)
    df = reduce(lambda x, y: merge(x, y, on_columns), df_generator)
    file_name_map = {
        f'{file}.Quantity': f'[{i + 1}] {file}.Quantity'
        for i, file in enumerate(quant.keys())
    }

    df.rename(columns=file_name_map, inplace=True)

    df.to_csv(save_path, index=False, sep='\t')


def cv_filter(x: pd.Series, cv: float):
    values = x.values[1:]
    return (values.std() / values.mean()) < cv

def get_sample_quant_fianlly(x: pd.Series, sample_columns: Sequence[str], is_A: bool):
    if is_A:
        return pd.Series([x[0], x[sample_columns].mean()], index=['strippedSequence', 'A'])
    else:
        return pd.Series([x[0], x[sample_columns].mean()], index=['strippedSequence', 'B'])

def get_quant_view_dataframe(df: pd.DataFrame, cv: float):
    """
        选出在所有 QC 样本均定量到的肽段, cv < 0.1/0.2

        Parameters:
        ---
        -   df: 定量得到的 dataframe
        -   cv: 筛选的阈值选择

        Returns:
        ---
        -   df_view: 筛选后的 dataframe
    """
    qc_file_nums = len(df.columns) - 1
    # print(qc_file_nums)
    start = 1
    # 索引左闭右开
    A_index_start, A_index_end = start, qc_file_nums // 2 + start
    B_index_start = (qc_file_nums + 1) // 2 + start
    A_samples, B_samples = df.columns[A_index_start:A_index_end].values, df.columns[B_index_start:].values
    # print(A_samples, B_samples)

    df_view = df.dropna(axis=0, how='any').reset_index(drop=True)
    # print(df_view.columns)

    df_view_group = df_view.groupby('strippedSequence')
    # 取 first 值
    view_quant_df = df_view_group.agg('first').reset_index(drop=False)
    cv_before_nums = len(view_quant_df)
    # print(cv_before_nums)
    cv_df = view_quant_df[view_quant_df.apply(cv_filter, axis=1, args=(cv, ))]
    cv_df = cv_df.reset_index(drop=True)
    A = cv_df.apply(get_sample_quant_fianlly, args=(A_samples, True, ), axis=1)
    B = cv_df.apply(get_sample_quant_fianlly, args=(B_samples, False, ), axis=1)
    # print(A.columns, B.columns)
    sample_df = pd.merge(left=A, right=B, how='inner')
    cv_after_nums = len(cv_df)
    return sample_df, cv_before_nums, cv_after_nums