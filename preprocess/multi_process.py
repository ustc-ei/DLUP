import os
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Callable
from .mergePeaks import multiProcess


def multi_process_mzml_tims(num_processes: int, root_path: str, extension_class: str, save_path: str, tol: int, func: Callable = multiProcess, *args):
    """
        多进程处理 mzml 文件
        ---
        提取 mzml 文件中质谱峰信息并进行峰合并操作

        Parameters:
        ---
        -   num_processes: 最大进程数
        -   root_path: mzml 文件所在文件夹
        -   extension_class: 文件后缀名 - '.mzML'/'.d'
        -   save_path: 峰合并操作后的数据存储的目标文件夹
        -   tol: 峰合并的阈值参数
        -   func: 多进程函数，一般为 `multiProcess` 函数
        -   agrs: func 的其他参数
    """
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []

        for path in [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(extension_class)]:
            future = executor.submit(
                func, path, save_path, tol, *args)
            futures.append(future)

        # 等待所有任务完成
        wait(futures)
    for f in futures:
        print(f.done())
    print("All processes are done.")


def main(root_path: str, num_processes: int = 20, tol: int = 15):
    save_path = os.path.join(root_path, 'merge')
    multi_process_mzml_tims(
        num_processes=num_processes,
        root_path=root_path,
        extension_class='.mzML',
        save_path=save_path,
        tol=tol,
        func=multiProcess
    )


if __name__ == "__main__":
    num_processes = 20
    import sys
    args = sys.argv
    tol = int(args[1])
    root_path = args[2]
    save_path = os.path.join(root_path, 'merge')
    # print(root_path)
    # print(save_path)
    # print(tol)
    # python3 multi_process.py 15 /data1/xp/data/astral_20231016_300ngPlasmaSample

    main(root_path, num_processes, tol)

    # multi_process_mzml_tims(
    #     num_processes=num_processes,
    #     root_path=root_path,
    #     extension_class='.mzML',
    #     save_path=save_path,
    #     tol=tol,
    #     func=multiProcess
    # )
