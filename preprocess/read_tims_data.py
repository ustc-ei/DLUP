import os
from tqdm import tqdm
from typing import Tuple, Dict
import time

import numpy as np
import numpy.typing as npt
from timspy.dia import TimsPyDIA

from .mergePeaks import mergePeaks
from preprocess.multi_process import multi_process_mzml_tims
from dia_dl.generatedata.utils.utils import create_dir


def concatPeaks(peaks: npt.NDArray,
                concatedPeak: Tuple[np.float64, np.float64]):
    """
        用于拼接同一个 scan 下的多个峰 

        尽管有很多不同的 frame, 但是每个 frame 下可能会有相同的 scan, 每个 scan 对应着一个峰, 并且他们有着同一个淌度信息, 因此我们把它们看作一个二级质谱

        >>> 峰信息就是拼接后的信息, 淌度就是这个 scan 的信息

        Input Parameters:
        ---
        -   `peaks`: 已经拼接完成的峰
        -   `concatedPeak`: 待拼接的峰
        ### Return:
        -   拼接完成的峰
    """
    return np.concatenate((
        peaks,
        np.array([
            concatedPeak
        ])
    ),
        axis=0)


def initialPeak(peak: Tuple[np.float64, np.float64]):
    """
        将峰信息转化为二维的 ndarray    

        ### Input Parameters:
        -   `peak`: 待初始化的峰
        ### Return:
        -   初始化后的峰
    """
    return np.array([
        peak
    ])


def extractMs2(path: str):
    """
        读取 `.d` 文件中的质谱数据, 将其中的二级质谱数据提取出来

        最后得到如下的数据格式
        [
            # 峰    扫描的窗口   保留时间 frame标号 scan标号  淌度信息
            [peaks, scanWindow, frameRT, frameID, scanID, ionMobity],
            ...
        ]

    """
    D = TimsPyDIA(path)
    winGroupScan: Dict[int, Dict[Tuple[int, int], Tuple]] = {}
    # [WindowGroup, ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth, CollisionEnergy, mz_left, mz_right]
    for wg in D.windows.values:
        if int(wg[0]) not in winGroupScan.keys():
            winGroupScan[int(wg[0])] = {
                (int(wg[1]), int(wg[2])): (wg[-2], wg[-1])
            }
        else:
            winGroupScan[int(wg[0])][(int(wg[1]), int(wg[2]))
                                     ] = (wg[-2], wg[-1])
    massSpectrums = []

    for frameID in D.ms2_frames:
        # [frame, scan, tof, intensity, mz, inv_ion_mobility, retention_time]
        ms2FrameData = D.query(frames=frameID)
        frameRT = ms2FrameData['retention_time'][0]
        # key: scanID, value: [peaks, inv_ion_mobility]
        scanDataDict = {}
        mzIndex = 4  # mz 在 dataframe 中对应的列下标
        intensityIndex = 3  # intensity 在 dataframe 中对应的下标
        ionMobityIndex = 5
        scanIdIndex = 1
        for scanData in ms2FrameData.values:
            scanID = int(scanData[scanIdIndex])
            if int(scanID) not in scanDataDict.keys():
                peak = initialPeak(
                    (scanData[mzIndex], scanData[intensityIndex])
                )
                scanDataDict[scanID] = [
                    peak,
                    scanData[ionMobityIndex]
                ]
            else:  # 将相同 scan 下的峰整合为一组峰
                scanDataDict[scanID][0] = concatPeaks(
                    scanDataDict[scanID][0],
                    (scanData[mzIndex], scanData[intensityIndex])
                )
        # end for scanData
        # 找到 frame 对应的 window
        # 如果 scanID 在 window 对应的 scan 范围内, 则将其二级质谱数据保留
        frameToWindow = (frameID - 1) % (max(winGroupScan.keys()) + 1)
        for scanID, massSpectrum in scanDataDict.items():
            peaks = massSpectrum[0]
            peaks = peaks[np.argsort(peaks[:, 0])]
            ionMobity = massSpectrum[1]
            for scanRange, scanWindow in winGroupScan[frameToWindow].items():
                if scanRange[0] <= scanID <= scanRange[1]:
                    massSpectrums.append(
                        [peaks, scanWindow, frameRT, frameID, scanID, ionMobity])
                    break

    array_massSpectrums = np.array(massSpectrums, dtype=object)
    del massSpectrums
    del D
    return array_massSpectrums


def processPeaks(massSpectrums: npt.NDArray,
                 file: str,
                 tol: int
                 ):
    """
        处理二级质谱的峰

        返回峰合并操作之后的质谱数据
    """
    for i in range(len(massSpectrums)):
        massSpectrums[i][0] = mergePeaks.mergePeaks(massSpectrums[i][0], tol)
    windows = np.array(list(set(massSpectrums[:, 1])))
    windows = windows[np.argsort(windows[:, 0])]
    windows = [tuple(window) for window in windows]
    print("divide the Ms2 by scan windows!")
    dividedMs2 = mergePeaks.divideMS2ByWindows(
        massSpectrums, windows)  # type: ignore
    return dividedMs2


def read_tims_data(filePath: str, rootPath: str, tol: int):
    """
        读取 tims 数据

        从 .d 文件数据中提取二级质谱数据并进行峰合并操作

        Parameters
        ---
        -   filePath: .d 文件所在文件路径
        -   root_path: 保存提取后的数据根目录
        -   
    """
    print("start extractint MassSpectrums!")
    massSpectrums = extractMs2(filePath)
    fileName = filePath.split('/')[-1].split('.')[0]
    savePath = os.path.join(rootPath, "extractMs2", f"{fileName}.npy")
    create_dir(os.path.dirname(savePath))
    np.save(savePath, massSpectrums)
    print("end!")

    peaksNumSatifyMassSpectrums = []
    for massSpectrum in massSpectrums:
        if len(massSpectrum[0]) >= 6:
            peaksNumSatifyMassSpectrums.append(massSpectrum)
    peaksNumSatifyMassSpectrums = np.array(
        peaksNumSatifyMassSpectrums, dtype=object)

    print("start merging Peaks!")
    afterMergedMs2 = processPeaks(
        peaksNumSatifyMassSpectrums, fileName, tol)
    savePath = os.path.join(rootPath, "merge", f"{fileName}.npy")
    create_dir(os.path.dirname(savePath))
    np.save(savePath, afterMergedMs2)  # type: ignore
    print("end!")
    del massSpectrums
    del peaksNumSatifyMassSpectrums
    del afterMergedMs2


def main(root_path: str, num_processes: int = 20, tol: int = 15):
    start = time.time()
    multi_process_mzml_tims(
        num_processes=num_processes,
        root_path=root_path,
        extension_class='.d',
        save_path=root_path,
        tol=tol,
        func=read_tims_data
    )
    end = time.time()
    t = end - start
    hour = t // 3600
    minute = (t - hour * 3600) // 60
    second = t - hour * 3600 - minute * 60
    print('{:.1f}h {:.1f}m {:.1f}s'.format(hour, minute, second))


if __name__ == "__main__":
    root_path = '/data/xp/data/ttp_20230706_H15150'
    start = time.time()
    main(root_path, 20, 15)
    end = time.time()
    t = end - start
    hour = t // 3600
    minute = (t - hour * 3600) // 60
    second = t - hour * 3600 - minute * 60
    print('{:.1f}h {:.1f}m {:.1f}s'.format(hour, minute, second))
    # import os
    # tol = 15
    # rootPath = "./"
    # filePaths = [rootPath +
    #              file for file in os.listdir(rootPath) if file.endswith('.d')]
    # for filePath in filePaths:
    #     print("start extractint MassSpectrums!")
    #     massSpectrums = extractMs2(filePath)
    #     fileName = filePath.split('/')[-1].split('.')[0]
    #     savePath = os.path.join(rootPath, "extractMs2", "{fileName}.npy")
    #     np.save(savePath, massSpectrums)
    #     print("end!")

    #     peaksNumSatifyMassSpectrums = []
    #     for massSpectrum in massSpectrums:
    #         if len(massSpectrum[0]) >= 6:
    #             peaksNumSatifyMassSpectrums.append(massSpectrum)
    #     peaksNumSatifyMassSpectrums = np.array(
    #         peaksNumSatifyMassSpectrums, dtype=object)

    #     print("start merging Peaks!")
    #     afterMergedMs2 = processPeaks(
    #         peaksNumSatifyMassSpectrums, fileName, tol)
    #     savePath = os.path.join(rootPath, "merge", "{fileName}.npy")
    #     np.save(savePath, afterMergedMs2)  # type: ignore
    #     print("end!")
