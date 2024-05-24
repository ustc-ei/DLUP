from typing import Sequence, Dict

from bidict import bidict
import numpy as np
import numpy.typing as npt
import pandas as pd


def get_only_value(hashDict: Dict, df: pd.DataFrame, columns: Sequence[str]):
    """
        选取第一个数值, 因为所有数都是相同的
    """
    for column in columns:
        hashDict[column] = df[column].values[0]


def fill_peaks(peaks: npt.NDArray, fragments: npt.NDArray):
    if len(peaks) < 6:
        peaks = np.concatenate((np.zeros((6-len(peaks), 2)), peaks), axis=0)
        fragments = np.concatenate((
            np.array([None, None, None, None] *
                     (6 - len(peaks))).reshape(-1, 4),
            fragments
        ), axis=0)
    return peaks, fragments


def selected_intensity_top6(peaks: npt.NDArray, fragments: npt.NDArray):
    """
        选取峰强度前六的峰以及其离子类型

        可能出现有多个相同的 b/y 离子出现在筛选的峰中
    """
    selected_indexs = np.argsort(-peaks[:, 1])[:6]
    peaks, fragments = peaks[selected_indexs], fragments[selected_indexs]
    sort_indexs = np.argsort(peaks[:, 0])
    peaks = peaks[sort_indexs]
    fragments = fragments[sort_indexs]
    peaks, fragments = fill_peaks(peaks, fragments)
    return peaks, fragments


def process_fragment(hashDict: Dict, df: pd.DataFrame, peak_columns: Sequence[str], fragment_columns: Sequence[str]):
    """
        将数据分为两部分

        1. 峰 [mz, intensity]

        2. 峰的类型以及原始形式 [type, number, lossType, charge] 
    """
    peaks = df[peak_columns].values
    fragments = df[fragment_columns].values
    # 如果无自定义的筛选函数, 则默认是选峰强度前六的峰
    peaks, fragments = selected_intensity_top6(peaks, fragments)
    # 存入字典中
    hashDict['Spectrum'] = peaks
    hashDict['Fragment'] = fragments


def target_library(
    raw_library_path: str,
):
    raw_library = pd.read_csv(raw_library_path, sep='\t', low_memory=False)
    group_columns = ['ModifiedPeptide', 'PrecursorCharge']
    #
    df = raw_library[['FragmentCharge', 'FragmentLossType', 'ExcludeFromAssay', 'ModifiedPeptide', 'StrippedPeptide',
                      'PrecursorCharge', 'iRT', 'IonMobility', 'PrecursorMz', 'FragmentNumber', 'FragmentType', 'FragmentMz', 'RelativeIntensity']]
    split_group = df.groupby(group_columns)
    # generate target library
    library = {}
    for key in split_group.groups.keys():
        modifiedPeptideProperty = split_group.get_group(key)
        library[key] = {}
        get_only_value(library[key], modifiedPeptideProperty, [
                       'StrippedPeptide', 'PrecursorMz', 'iRT', 'IonMobility'])
        process_fragment(library[key], modifiedPeptideProperty, ['FragmentMz', 'RelativeIntensity'], [
                         'FragmentType', 'FragmentNumber', 'FragmentLossType', 'FragmentCharge'])
    return library


def decoy_library(
    target: Dict,
    distance: int
):
    DecoyLibrary = {}
    charges = set(np.array(list(target.keys()))[:, 1].astype(int))
    for charge in charges:
        LibraryDividedByCharge = {key: value for key,
                                  value in target.items() if key[1] == charge}
        DecoyLibraryDividedByCharge = {}
        swapped_keys = bidict()   # 双向字典, 记录互相交换的2个母离子

        key_premz = []
        for key, value in LibraryDividedByCharge.items():
            key_premz.append([key, value['PrecursorMz']])
        key_premz = np.array(key_premz)
        key_premz = key_premz[np.argsort(key_premz[:, 1])]  # 按母离子MZ排序

        i, j = 0, 1
        while i < len(key_premz) and j < len(key_premz):
            i_key, i_premz, j_key, j_premz = key_premz[i][0], key_premz[i][1], key_premz[j][0], key_premz[j][1]
            if abs(j_premz-i_premz) >= distance:
                decoy_i_key = ('DECOY-'+i_key[0], i_key[1])
                decoy_j_key = ('DECOY-'+j_key[0], j_key[1])

                DecoyLibraryDividedByCharge[decoy_i_key] = target[i_key].copy()
                DecoyLibraryDividedByCharge[decoy_i_key]['Spectrum'] = target[j_key]['Spectrum'].copy(
                )

                DecoyLibraryDividedByCharge[decoy_j_key] = target[j_key].copy()
                DecoyLibraryDividedByCharge[decoy_j_key]['Spectrum'] = target[i_key]['Spectrum'].copy(
                )

                # 峰移
                delta = i_premz - j_premz
                DecoyLibraryDividedByCharge[decoy_i_key]['Spectrum'][:, 0] += delta
                DecoyLibraryDividedByCharge[decoy_j_key]['Spectrum'][:, 0] -= delta

                swapped_keys[i_key] = j_key
            else:
                j += 1

            while (i < len(key_premz)) and (key_premz[i][0] in (set(swapped_keys) | set(swapped_keys.inv))):
                i += 1
            while (j < len(key_premz)) and (key_premz[j][0] in (set(swapped_keys) | set(swapped_keys.inv))):
                j += 1

        if len(LibraryDividedByCharge) != len(DecoyLibraryDividedByCharge):   # 还有没交换的
            unswapped_keys = set(LibraryDividedByCharge.keys(
            )) - (set(swapped_keys) | set(swapped_keys.inv))  # 差集
            unswapped_keys = list(unswapped_keys)
            unswapped_keys.sort()

            for unswapped_key in unswapped_keys:
                unswapped_premz = target[unswapped_key]['PrecursorMz']
                for swapped_key in sorted(swapped_keys):
                    swapped_key2 = swapped_keys[swapped_key]
                    if abs(unswapped_premz - target[swapped_key]['PrecursorMz']) >= distance and \
                       abs(unswapped_premz - target[swapped_key2]['PrecursorMz']) >= distance:  # 三个母离子,两两的M/Z差>distance
                        DecoyLibraryDividedByCharge[(
                            'DECOY-'+unswapped_key[0], unswapped_key[1])] = target[unswapped_key].copy()
                        DecoyLibraryDividedByCharge[(
                            'DECOY-'+unswapped_key[0], unswapped_key[1])]['Spectrum'] = target[swapped_key2]['Spectrum'].copy()

                        DecoyLibraryDividedByCharge[(
                            'DECOY-'+swapped_key[0], swapped_key[1])]['Spectrum'] = target[unswapped_key]['Spectrum'].copy()

                        # 峰移
                        DecoyLibraryDividedByCharge[('DECOY-'+swapped_key[0], swapped_key[1])]['Spectrum'][:, 0] += (
                            target[swapped_key]['PrecursorMz']-unswapped_premz)
                        DecoyLibraryDividedByCharge[('DECOY-'+unswapped_key[0], unswapped_key[1])]['Spectrum'][:, 0] += (
                            target[unswapped_key]['PrecursorMz']-target[swapped_key2]['PrecursorMz'])

                        break
                if swapped_key in swapped_keys:
                    swapped_keys.pop(swapped_key)  # 避免重复交换
        DecoyLibrary.update(DecoyLibraryDividedByCharge)
    return DecoyLibrary


def process_intensity(library: Dict):
    for _, metadata in library.items():
        metadata['Spectrum'][:, 1] = metadata['Spectrum'][:, 1] / \
            np.sum(metadata['Spectrum'][:, 1])

    return library


def main(raw_library_path: str):
    print('-'*40)
    print(' ' * 5, 'target library is generating!', ' ' * 5)
    print('-'*40)
    target = target_library(raw_library_path)
    print('-'*40)
    print(' ' * 5, 'decoy library is generating!', ' ' * 5)
    print('-'*40)
    decoy = decoy_library(target, 100)
    #
    target = process_intensity(target)
    decoy = process_intensity(decoy)
    return target, decoy
