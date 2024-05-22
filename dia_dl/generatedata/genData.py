import os
import time
from typing import Dict, List, Any, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, wait

import torch

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .utils.calculateFunctions import (
    calculateMassSpectrumMzTranverse,
)

from .utils.dataPretreatment import ionMobilityPretreatment

from .utils.peaksProcessFunctions import (
    divideLibraryByWindows,
    peaksMatch,
    fillMassSpectrumWithZeros,
    fillMassSpectrumIonMobilityWithZeros
)
from .utils.utils import read_dict_npy, create_dir
from .utils import collection


def fill_dict_value(d, key, value):
    d[key] = value


def initial_dict_info(d, key):
    """
        初始化字典中的 value

        初始化 value 为空的字典
    """
    d[key] = {}


def quant_train(
    peptide_label_info: Dict,
    peptideNameWithCharge: Tuple[str, int],
    labels: Dict[str, Any],
):
    if peptideNameWithCharge in labels:
        peptide_label_info['label'] = labels[peptideNameWithCharge]
        return 1
    return 0


def quant_test(peptide_label_info):
    peptide_label_info['label'] = 0
    return 1


def identify_train(
    peptide_label_info: Dict,
    peptideNameWithCharge: Tuple[str, int],
    labels: Dict[str, Any],
):
    if peptideNameWithCharge in labels:
        peptide_label_info['label'] = 1
        return 1
    return 0


def identify_test(
    peptide_label_info: Dict,
):
    peptide_label_info['label'] = 0
    return 1


def identify_decoy(
    peptide_label_info: Dict,
):
    peptide_label_info['label'] = 0
    return 1


<<<<<<< HEAD
def calculate_peaks_sum(
    filter_intensity: npt.NDArray
):
    peaks_sum = np.sum(filter_intensity, axis=1)
    max_value, min_value = np.max(peaks_sum), np.min(peaks_sum)
    return (peaks_sum - min_value) / (max_value - min_value)

=======
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
def calculate_ms2distance(
    filter_intensity: npt.NDArray,
    spectrum_intensity: npt.NDArray
):
    intensity_sum = np.sum(
        filter_intensity, axis=1, keepdims=True) + 1e-7
    normolize = filter_intensity / intensity_sum

    normolize[normolize == 0.0] = 1.0

    distance = np.abs(
        spectrum_intensity - normolize)
    distance_sum = np.sum(distance, axis=1)

    return distance_sum


def filter(
    file: str,
    peptide_label_info: Dict[str, npt.NDArray],
    peptideName: str,
    charge: int,
    configs: Dict
):
    """
        Filters and processes peptide label information based on specified conditions.

        Parameters:
        ---
        - file: Filename, a string specifying the path to the data file.
        - peptide_label_info: A dictionary containing detailed peptide label information, which should include key arrays like Ms2IonMobility and CandidateMs2.
        - peptideName: Peptide name, a string used to identify a specific peptide.
        - charge: Charge state, an integer representing the charge status of the peptide.
        - configs: A dictionary with various configuration options, such as whether to consider ion mobility (is_ionMobility) and the number of spectra to filter (filterMassSpectrumNums).

        Returns:
        ---
        None. The function modifies the input `peptide_label_info` dictionary and saves it to a designated path.
    """
    if 'label' not in peptide_label_info:
        return
    if configs['is_ionMobility']:
        ionMobilityDistance = np.abs(
            peptide_label_info['Ms2IonMobility'] - peptide_label_info['IonMobility'])

        # 筛选出淌度差值小于设定阈值的实验图谱对应下标
        selectedIndex = (ionMobilityDistance <
                         configs['mobilityDistanceThreshold']).nonzero()
        # 如果它匹配的所有的图谱淌度差都大于阈值, 则我们只留下淌度差值最小的, 再进行填充
        if len(selectedIndex) == 0:
            index = np.argmin(ionMobilityDistance)
            peptide_label_info['matchedMs2'] = peptide_label_info['CandidateMs2'][index]
            peptide_label_info['Ms2IonMobility'] = np.array(
                peptide_label_info['Ms2IonMobility'][index])
        else:
            peptide_label_info['matchedMs2'] = peptide_label_info['CandidateMs2'][selectedIndex]
            peptide_label_info['Ms2IonMobility'] = peptide_label_info['Ms2IonMobility'][selectedIndex]

        peptide_label_info['Ms2IonMobility'] = fillMassSpectrumIonMobilityWithZeros(
            filterMassSpectrumNums=configs['filterMassSpectrumNums'],
            massSpectrumsIonMobility=peptide_label_info['Ms2IonMobility']
        )

    if 'matchedMs2' not in peptide_label_info:
        peptide_label_info['matchedMs2'] = peptide_label_info['CandidateMs2']

    peptide_label_info['matchedMs2'] = fillMassSpectrumWithZeros(
        massSpectrumsPeaks=peptide_label_info['matchedMs2'],
        filterMassSpectrumNums=configs['filterMassSpectrumNums'],
        peptidePeakNums=configs['peptidePeakNums']
    )

    peptideMatchedMassSpectrumsPeakIntensity = peptide_label_info[
        'matchedMs2'][:, :, 1]
    peptideMassSpectrumPeakIntensity = peptide_label_info['Spectrum'][:, 1]

    intensityDistanceSum = calculate_ms2distance(
        filter_intensity=peptideMatchedMassSpectrumsPeakIntensity,
        spectrum_intensity=peptideMassSpectrumPeakIntensity,
    )

<<<<<<< HEAD
    filter_features = intensityDistanceSum

    if configs['is_quant']:
        peaks_sum = calculate_peaks_sum(peptideMatchedMassSpectrumsPeakIntensity)
        filter_features = intensityDistanceSum - peaks_sum

    sortIndex = np.argsort(filter_features)
=======
    sortIndex = np.argsort(intensityDistanceSum)
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    filterIndex = sortIndex[:configs['filterMassSpectrumNums']]
    filterIndex = np.sort(filterIndex)
    peptide_label_info['matchedMs2'] = peptide_label_info['matchedMs2'][filterIndex]
    if configs['is_ionMobility']:
        peptide_label_info['Ms2IonMobility'] = peptide_label_info['Ms2IonMobility'][filterIndex]
        peptide_label_info['Ms2IonMobility'] = np.array(
            peptide_label_info['Ms2IonMobility'], dtype=np.float32)
<<<<<<< HEAD
        
        peptide_label_info['Ms2IonMobility'] = ionMobilityPretreatment(
            peptide_label_info['Ms2IonMobility'], configs['peptidePeakNums'])
=======
        # print(peptide_label_info['Ms2IonMobility'])
        peptide_label_info['Ms2IonMobility'] = ionMobilityPretreatment(
            peptide_label_info['Ms2IonMobility'], configs['peptidePeakNums'])
        # print(peptide_label_info['Ms2IonMobility'])
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79

        peptide_label_info['IonMobility'] = np.array(
            peptide_label_info['IonMobility'], dtype=np.float32)
        peptide_label_info['IonMobility'] = np.repeat(
            peptide_label_info['IonMobility'],
            repeats=6, axis=0).reshape(1, 6)

    peptide_label_info['matchedMs2'] = np.array(
        peptide_label_info['matchedMs2'], dtype=np.float32)
    peptide_label_info['Spectrum'] = np.array(
        peptide_label_info['Spectrum'], dtype=np.float32)
    peptide_label_info['Spectrum'] = peptide_label_info['Spectrum'].reshape(
        (1, 6, 2))
    peptide_label_info['label'] = np.array(
        peptide_label_info['label'], dtype=np.float32)

    del peptide_label_info['CandidateMs2']

    save_path = os.path.join(configs['save_root_path'], file,
                             f'({peptideName}, {charge}).npy')

    create_dir(os.path.dirname(save_path))
    np.save(save_path, peptide_label_info)  # type: ignore
    del peptide_label_info


def match(
    file: str,
    window: Tuple,
    peptideNameWithCharge: Tuple,
    peptideInfo: Dict,
    massSpectrums: npt.NDArray,
    mzAfterTranverseList: npt.NDArray,
    labels: Dict,
    configs: Dict
):
    """
        Parameters:
        ---
        - `file`: Name of the file being processed.
        - `window`: A tuple specifying the range of spectra to be processed.
        - `peptideNameWithCharge`: A tuple containing the peptide name and its charge state.
        - `peptideInfo`: A dictionary containing information corresponding to the peptide from the spectral library, including spectra, etc.
        - `massSpectrums`: An array of experimental spectrum information.
        - `mzAfterTranverseList`: An array of processed results after shifting peaks in the experimental spectrum left and right.
        - `labels`: A dictionary of labels used to annotate the dataset categories.
        - `configs`: A dictionary of configuration information including various runtime parameters.

        Return:
        ---
        No direct return value, but processes the input parameters and calls other functions for further operations.
    """
    # match
    peptideName, charge = peptideNameWithCharge
    candidateMs2 = []  # 存储肽段匹配成功的实验图谱
    # rawMs2 = []  # 存储原始匹配的峰
    if configs['is_ionMobility']:
        candidateMs2IonMobility = []  # 存储肽段匹配成功的实验图谱的离子淌度 IonMobility
    peptideMassSpectrum = peptideInfo["Spectrum"]

    for ms2_i, ms2 in enumerate(massSpectrums[window]):
        matchedPeaks = []  # 存储匹配的峰
        massSpectrumPeaks = ms2[0]
        mzAfterTranverse = mzAfterTranverseList[ms2_i]
        insertIndex, peakMatchedNum = peaksMatch(
            peptideMassSpectrum[:, 0], mzAfterTranverse)
        if peakMatchedNum < configs['peptideMatchMs2PeakNums']:
            continue
        for i, index in enumerate(insertIndex):
            if index == -1:
                matchedPeaks.append([0.0, 0.0])
            else:
                matchedPeaks.append(
                    [peptideMassSpectrum[i, 0], massSpectrumPeaks[index, 1]])
        # rawMs2.append(massSpectrumPeaks)
        candidateMs2.append(matchedPeaks)
        if configs['is_ionMobility']:
            candidateMs2IonMobility.append(ms2[-1])
    # end for massSpectrums
    if len(candidateMs2) == 0:  # 一张实验图谱都没有匹配上
        return

    peptide_label_info = {
        'CandidateMs2': np.array(candidateMs2),
        'Spectrum': peptideMassSpectrum,
        'MatchedNum': len(candidateMs2),
        # 'rawMs2': np.array(rawMs2, dtype=object)
    }

    if configs['is_ionMobility']:
        fill_dict_value(
            peptide_label_info,
            'Ms2IonMobility',
            np.array(candidateMs2IonMobility)
        )
        fill_dict_value(
            peptide_label_info,
            'IonMobility',
            peptideInfo["IonMobility"]
        )
    if configs['is_featuredIons']:
        fill_dict_value(
            peptide_label_info,
            'FeaturedIons',
            peptideInfo["FeaturedIons"]
        )
    if configs['is_quant'] and not configs['is_test']:
        quant_train(
            peptide_label_info=peptide_label_info,
            peptideNameWithCharge=peptideNameWithCharge,
            labels=labels[file]
        )
    elif configs['is_quant'] and configs['is_test']:
        quant_test(peptide_label_info=peptide_label_info)

    elif configs['is_decoy']:
        identify_decoy(peptide_label_info=peptide_label_info)
    else:
        if configs['is_test']:
            identify_test(
                peptide_label_info=peptide_label_info)
        else:
            identify_train(
                peptide_label_info=peptide_label_info,
                peptideNameWithCharge=peptideNameWithCharge,
                labels=labels[file]
            )
    # filter
    filter(file, peptide_label_info, peptideName, charge, configs)
    del candidateMs2
    if configs['is_ionMobility']:
        del candidateMs2IonMobility


def window_match(
    window,
    file,
    massSpectrums,
    delta,
    labels,
    configs,
    library,
):
    """
        Within a specified window, this function matches experimental mass spectra with library spectra.

        Parameters:
        ---
        -   window: Window index, used to specify the start and end positions within the sequence of mass spectra.
        -   file: Filename, representing the current mass spectral data file being processed.
        -   massSpectrums: A sequence of mass spectra, containing information from a series of mass spectrometry runs.
        -   delta: Offset value, applied to adjust the mass-to-charge ratios of the mass spectra.
        -   labels: A sequence of labels, used to annotate additional information for each mass spectrum.
        -   configs: Configuration information, including algorithm parameters, etc.
        -   library: A spectral library storing known spectra of peptide segments.

        Returns: 
        ---
        None. 

        The primary role of this function is to invoke the match function for spectrum matching.
    """
    mzAfterTranverseList = []  # 存储每张质谱图左右偏移后的质荷比序列

    # 先将每张实验图谱的质荷比进行左右偏移, 将其存入 mzForMassSpectrum 中
    for ms2 in massSpectrums[window]:
        mzAfterTranverse = calculateMassSpectrumMzTranverse(
            ms2[0], delta)
        mzAfterTranverseList.append(mzAfterTranverse)

    for peptideNameWithCharge, peptideInfo in library[window].items():
        match(
            file=file,
            window=window,
            peptideNameWithCharge=peptideNameWithCharge,
            peptideInfo=peptideInfo,
            massSpectrums=massSpectrums,
            mzAfterTranverseList=mzAfterTranverseList,  # type: ignore
            labels=labels,
            configs=configs
        )
    del mzAfterTranverseList


def multi_process_match_ms(
    msFilePath: str,
    libraryPath: str,
    configs: Dict,
    **kwargs
):
    """
        Perform spectrum matching using multiple processes across windows.

        Parameters:
        ---
        - msFilePath: str, the path to the numpy binary file storing the spectrum data.
        - libraryPath: str, the path to the library containing spectra.
        - configs: Dict, configuration options required for the matching process, such as whether to use feature ions.
        - **kwargs: Additional keyword arguments for passing to the `window_match` function.

    """
    fileName = msFilePath.split('.npy')[0].split('/')[-1]
    print(f'read {fileName} data')
    massSpectrums = read_dict_npy(msFilePath)
    print('end')
    s_window = massSpectrums.keys()
    print("divide library by windows!")
    library = divideLibraryByWindows(
        libraryPath, s_window, configs['is_featuredIons'])
    print("end!")
    for window in s_window:
        window_match(window=window, file=fileName,
                     configs=configs, library=library,
                     massSpectrums=massSpectrums, **kwargs)
    del massSpectrums
    del library


def main(
    libraryPath: str,
    massSpectrumFilePathList: List[str],
    labels: Dict[str, Any],
    configs: Dict,
    func=multi_process_match_ms
):
    """
        The main function, responsible for processing spectrum matching and collection.

        Parameters:
        ---
        - libraryPath: Spectrum library path, a string type used to specify the location of the spectrum library.
        - massSpectrumFilePathList: List of spectrum file paths, a list of strings containing the paths of all spectrum files to be processed.
        - labels: Label information, a dictionary with keys as strings and values of any type, used to store various label information related to the spectra.
        - configs: Configuration information, a dictionary containing various configurations needed during execution, such as tolerance error and number of processes.
        - func: Parallel processing function, a callable object, defaulting to `multi_process_match_ms`, used to parallelize the spectrum matching task.

        No return value.
    """
    delta = configs['tol'] * 1e-6
    start = time.time()

    with ProcessPoolExecutor(max_workers=configs['num_processes']) as executor:
        futures = []
        for msFilePath in massSpectrumFilePathList:
            future = executor.submit(
                func,
                msFilePath=msFilePath,
                libraryPath=libraryPath,
                delta=delta,
                labels=labels,
                configs=configs
            )
            futures.append(future)
        # 等待所有任务完成
        wait(futures)
    print("All processes are done.")
    # only one process

    # for msFilePath in massSpectrumFilePathList:
    #     func(
    #         msFilePath=msFilePath,
    #         libraryPath=libraryPath,
    #         delta=delta,
    #         labels=labels,
    #         configs=configs
    #     )
    end = time.time()
    t = end - start
    hour = t // 3600
    minute = (t - hour * 3600) // 60
    second = t - hour * 3600 - minute * 60
    print('{} hour {} minute {} second'.format(hour, minute,  second))

    start = time.time()
    collection.main(80, configs['save_root_path'], configs['features'])
    end = time.time()
    t = end - start
    hour = t // 3600
    minute = (t - hour * 3600) // 60
    second = t - hour * 3600 - minute * 60
    print('collection: {} hour {} minute {} second'.format(hour, minute,  second))
