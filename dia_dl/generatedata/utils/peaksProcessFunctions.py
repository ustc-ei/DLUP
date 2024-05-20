import time
from typing import Dict, List, Any, Tuple
from numba import jit
import numpy as np
import numpy.typing as npt


def unique(x: npt.NDArray):
    """
        返回无重复元素的序列
    """
    unique_arr, counts = np.unique(x, return_counts=True)
    return unique_arr[counts == 1]


@jit(nopython=True)
def tranverse(peptideSpectrumMzs: npt.NDArray, tol: int):
    """
        寻找特异峰

        主要是将所有肽段的 mz 进行 tol 长度的偏移, 偏移的过程中将其记录其是否合并的标志位

        最后筛选出没有进行合并的 mz 就是特异峰离子的 mz
    """
    delta = tol * 1e-6
    featured_ions_flags = np.array(
        [True for _ in range(peptideSpectrumMzs.shape[0])])

    while True:
        # print(len(featured_ions_flags))
        mergeSpectrumMzsInsertIndex = np.searchsorted(
            peptideSpectrumMzs + delta * peptideSpectrumMzs,
            peptideSpectrumMzs,
            side='left')
        a = np.unique(mergeSpectrumMzsInsertIndex)
        if len(a) == len(peptideSpectrumMzs):
            break
        featured_ions_flags = featured_ions_flags[a]
        peptideSpectrumMzsAfterMerge = peptideSpectrumMzs[a]
        for i, seq_index in enumerate(a):
            n = 0
            for index in mergeSpectrumMzsInsertIndex:
                if index == seq_index:
                    n += 1
            if n > 1:
                featured_ions_flags[i] = False
        peptideSpectrumMzs = peptideSpectrumMzsAfterMerge
    return featured_ions_flags, peptideSpectrumMzs


def cal_featured_ions(library: Dict):
    featured_ions_num = 0
    for info in library.values():
        featured_ions_num += len(np.where(info['FeaturedIons'] == 1)[0])
    return len(library), featured_ions_num


def search_featured_ions(library_splitby_window: Dict):
    peptideSpectrumMzs = np.zeros((1, ))
    for _, info in library_splitby_window.items():
        peptideSpectrumMzs = np.concatenate(
            (peptideSpectrumMzs, info['Spectrum'][:, 0]), axis=0)

    peptideSpectrumMzs = unique(peptideSpectrumMzs)
    # print("Merge Before")
    # print(len(peptideSpectrumMzs))
    # start = time.time()
    featured_flags, peptideSpectrumMzsAfterMerge = tranverse(
        peptideSpectrumMzs, 15)
    # print("Merge After")
    # print(len(peptideSpectrumMzsAfterMerge))
    # print(len(featured_flags))
    # featured ions
    peptideSpectrumMzs = peptideSpectrumMzsAfterMerge[featured_flags]
    # print('Featured ions')
    # print(peptideSpectrumMzs)
    for _, info in library_splitby_window.items():
        mz = info['Spectrum'][:, 0]
        featured_ions = np.zeros_like(mz)
        featured_ions_index = np.where(np.in1d(mz, peptideSpectrumMzs))[0]
        featured_ions[featured_ions_index] = 1
        info['FeaturedIons'] = featured_ions
    # print(library_splitby_window)
    # cal_featured_ions(library_splitby_window)
    # end = time.time()
    # print((end - start), 's')
    # print(featured_flags)
    # print(peptideSpectrumMzsAfterMerge)
    # print("After Merge")
    # print(len(peptideSpectrumMzsAfterMerge))


def cal_featured_ions_library(library: Dict):
    modified_peptide_sum, featured_ions_sum = 0, 0
    for _, library_split_by_window in library.items():
        modified_peptide_num, featured_ions_num = cal_featured_ions(
            library_split_by_window)
        modified_peptide_sum += modified_peptide_num
        featured_ions_sum += featured_ions_num
    # print('modified_peptide_num: ', modified_peptide_sum)
    # print('featured ions num: ', featured_ions_sum)


def divideLibraryByWindows(
    spectraLibraryPath: str,
    windows: List[Tuple[int, ...]],
    if_featured_ions: bool
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
        将图谱库中的肽段根据窗口 scan 的范围进行划分

        图谱库中包含了肽段的前体离子的 MZ, MZ 在扫描碎裂的窗口的 MZ 范围内说明肽段的前体离子在该窗口内进行碎裂

        因为我们预处理的质谱数据也是根据窗口进行划分的, 这样做方便我们后续进行图谱和肽段的匹配操作

        ### Input Parameters:
        -   spectraLibraryPath: 图谱库的文件路径
        -   windows: 质谱文件划分的窗口

        ### Return:
        -   dividedLibrary: 经质谱数据窗口划分后的图谱库
    """
    library = np.load(spectraLibraryPath, allow_pickle=True).item()
    dividedLibrary = {}
    for window in windows:
        dividedLibrary[window] = {
            key: value
            for key, value in library.items()
            if window[0] <= value["PrecursorMZ"] < window[1]
        }
    if if_featured_ions:
        for window in windows:
            search_featured_ions(dividedLibrary[window])
        cal_featured_ions_library(dividedLibrary)
    return dividedLibrary


@jit(nopython=True)
def searchInsertIndex(a: np.ndarray, b: np.ndarray):
    """
    将 b 中每个元素插入到 a 中, 并返回插入的下标序列

    > (可以设置元素相等时插入前面还是后面, `side = left/right` 控制)

    ### 这个函数适用于 `峰匹配操作`

    -   通常 `a` 为进行左右 `tol * mz` 偏移后的实验图谱峰的质荷比序列 
    -   (每张实验图谱我们在预处理阶段已经进行了峰合并操作, 因此左右偏移时必然不会出现 |mz1 - mz2| < mz1 * tol 的情况)
    -   `b` 为肽段的理论图谱峰的质荷比序列

    -   如果插入的下标满足 `index % 2 == 1`, 则说明峰匹配成功, 我们还可以获取匹配了原始图谱的第几个峰 (`(index - 1)// 2`)

    -   示例: `1, 2, 3, 4, 5`, `tol = 0.05` (数据不一定真实存在, 只是为了方便说明而随意设计的)
    1. `input`: `a` = `np.array([0.95, 1.05, 1.9, 2.1, 2.85, 3.15, 3.8, 4.75, 5.25])`, `b` = `np.array([1.02, 1.8, 2.9])`
    2. `indexs = [1, 2, 5]`, 注意: `2` 表示没有匹配上, 其余对应匹配了原始图谱的 `[0, 2]` 峰 
    3. `return indexs`


    ### Input Parameters:
    -   `a`: 被插的 array
    -   `b`: 需要插入的 array

    ### Return:
    -   `indexs`: 返回的插入下标序列 
    """
    return np.searchsorted(a, b)


def peaksMatch(peptidePeaksMz: np.ndarray, massSpectrumPeaksMz: np.ndarray):
    """
    ### 峰匹配函数

    |mz1 - mz2| < delta * mz1 则表示峰匹配成功, 我们需要找到肽段参考图谱和实验图谱峰匹配成功对应于实验图谱的下标

    直接调用 `searchInsertIndex` 函数可以返回下标

    获得它在实验图谱中的下标可以见 `searchInsertIndex` 的注释部分, 写的非常清楚了

    没有匹配成功的部分直接就补 0

    ### Input Parameters:
    -   peptidePeaksMz: 肽段实验图谱的质荷比序列
    -   massSpectrumPeaksMz: 实验图谱的质荷比序列 (左右偏移后的)
    ### Return:
    -   criticalIndexs: 峰匹配的下标序列 (成功则对应于原始实验图谱质谱的下标, 失败则是 -1)
    -   peakMatchedNum: 峰匹配成功的个数
    """
    insertIndexs = searchInsertIndex(massSpectrumPeaksMz, peptidePeaksMz)
    criticalIndexs = []
    peakMatchedNum = 0
    for index in insertIndexs:
        if index % 2 == 1:
            criticalIndexs.append((index - 1) // 2)
            peakMatchedNum += 1
        else:
            criticalIndexs.append(-1)
    return criticalIndexs, peakMatchedNum


def fillMassSpectrumIonMobilityWithZeros(
    filterMassSpectrumNums: int,
    massSpectrumsIonMobility: npt.NDArray,
):
    peptideMatchMs2Num = len(massSpectrumsIonMobility)
    filledMassSpectrumsIonMobility = massSpectrumsIonMobility

    if peptideMatchMs2Num < filterMassSpectrumNums:
        filledMassSpectrumsIonMobility = np.concatenate((
            massSpectrumsIonMobility,
            np.zeros(filterMassSpectrumNums - peptideMatchMs2Num)
        ), axis=0)

    return filledMassSpectrumsIonMobility


def fillMassSpectrumWithZeros(
    massSpectrumsPeaks: npt.NDArray,
    filterMassSpectrumNums: int,
    peptidePeakNums: int,
) -> npt.NDArray:
    peptideMatchMs2Num = len(massSpectrumsPeaks)
    filledMassSpectrumsPeaks = massSpectrumsPeaks

    if peptideMatchMs2Num < filterMassSpectrumNums:
        filledMassSpectrumsPeaks = np.concatenate((
            massSpectrumsPeaks,
            np.full((filterMassSpectrumNums -
                     peptideMatchMs2Num, peptidePeakNums, 2), 0)
        ), axis=0)

    return filledMassSpectrumsPeaks
