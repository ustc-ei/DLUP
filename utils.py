import os
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np


def collection_file(folder: str, features: Sequence[str]):
    data = {
        feature: []
        for feature in features
    }
    for path in os.listdir(folder):
        if os.path.exists(os.path.join(folder, 'collection.npy')):
            os.remove(os.path.join(folder, 'collection.npy'))
        f = os.path.join(folder, path)
        d = np.load(f, allow_pickle=True).item()
        modified_charge = path.split('/')[-1].split('.npy')[0][1:-1]
        modified_charge = modified_charge.split(', ')
        modified, charge = modified_charge[0], modified_charge[1]
        d['info'] = np.array((dir, (modified, int(charge))), dtype=object)
        for feature in features:
            data[feature].append(d[feature])
    for feature in features:
        data[feature] = np.stack(data[feature], axis=0)
    np.save(os.path.join(folder, 'collection.npy'), data)


def collection_dir(root: str, features: Sequence[str]):
    data = {
        feature: []
        for feature in features
    }

    for dir in os.listdir(root):
        folder = os.path.join(root, dir)
        if not os.path.isdir(folder):
            continue
        path = os.path.join(folder, 'collection.npy')
        collection = np.load(path, allow_pickle=True).item()
        for f in features:
            data[f].append(collection[f])
    for f in features:
        data[f] = np.concatenate(data[f], axis=0)
    np.save(os.path.join(root, 'collection.npy'), data)


def main(num_processes: int, root: str, features: Sequence[str]):
    print("Start!")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for dir in os.listdir(root):
            folder = os.path.join(root, dir)
            future = executor.submit(
                collection_file,
                folder=folder,
                features=features
            )
            futures.append(future)
        # 等待所有任务完成
        wait(futures)
    print("All processes are done.")
    collection_dir(root, features)
    print("End!")
