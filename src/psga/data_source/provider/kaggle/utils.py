import random
from collections import defaultdict
from typing import (
    Tuple,
    List,
    Iterable
)

import cv2
import numpy as np
from numpy.random import choice


def zoom_tiles(tiles: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return np.asarray([cv2.resize(tile, dsize=shape, interpolation=cv2.INTER_LANCZOS4) for tile in tiles])


def bin_label(label: int, classes: int) -> np.ndarray:
    binning = np.zeros(shape=(classes - 1))
    binning[:label] = 1
    return binning


def balanced_subsample(sequence: Iterable, count: int) -> List[int]:
    count_per_label = int(np.ceil(count / np.unique(sequence).shape[0]))
    index_map = defaultdict(list)
    [index_map[label].append(index) for index, label in enumerate(sequence)]
    subsample_indices = list()

    for indices in sorted(index_map.values(), key=lambda x: len(x)):
        if len(indices) < count_per_label:
            subsample_indices.extend(indices)
        else:
            subsample_indices.extend(choice(indices, size=count_per_label, replace=False))

    unused_indices = list(set(range(len(sequence))) - set(subsample_indices))
    absent = count - len(subsample_indices)
    if absent > 0:
        subsample_indices.extend(choice(unused_indices, size=absent, replace=False))
    random.shuffle(subsample_indices)

    return subsample_indices
