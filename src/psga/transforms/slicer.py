from functools import reduce
from typing import (
    Optional,
    NoReturn,
    Tuple
)

import numpy as np


class TilesSlicer(object):

    def __init__(self, tile_size: int, intersection: float = 0.0, fill_value: int = 255,
                 remove_empty_tiles: bool = False, emptiness_degree: float = 1.0) -> NoReturn:
        assert 0 <= intersection <= 1
        self._tile_size = tile_size
        self._intersection = intersection
        self._step = int(tile_size * intersection)
        self._fill_value = fill_value

        self._remove_empty_tiles = remove_empty_tiles
        self._emptiness_degree = emptiness_degree

    def __call__(self, image: np.ndarray, non_empty_tiles_indices: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        if self._intersection == 0:
            tiles = self._fast_non_overlapped_slice(image)
        else:
            tiles = self._slow_overlapped_slice(image)

        if self._remove_empty_tiles:
            if non_empty_tiles_indices is None:
                empty_sum = reduce(lambda x, y: x * y, tiles.shape[1:]) * self._fill_value * self._emptiness_degree
                non_empty_tiles_indices = np.where(tiles.sum(axis=(1, 2, 3)) < empty_sum)[0]
            tiles = tiles[non_empty_tiles_indices]

        return tiles, non_empty_tiles_indices

    def _fast_non_overlapped_slice(self, image: np.ndarray) -> np.ndarray:
        pad_height = self._tile_size - image.shape[0] % self._tile_size
        pad_width = self._tile_size - image.shape[1] % self._tile_size
        if pad_height != self._tile_size or pad_width != self._tile_size:
            pad_height = 0 if pad_height == self._tile_size else pad_height
            pad_width = 0 if pad_width == self._tile_size else pad_width
            image = self._pad_image(image, pad_width, pad_height)

        n_tiles_h = image.shape[0] // self._tile_size
        n_tiles_w = image.shape[1] // self._tile_size
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        tiles = image.reshape((n_tiles_h, self._tile_size, n_tiles_w, self._tile_size, channels))
        tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(-1, self._tile_size, self._tile_size, channels)
        tiles = np.squeeze(tiles)

        return tiles

    def _slow_overlapped_slice(self, image: np.ndarray) -> np.ndarray:
        x_steps = np.arange(0, image.shape[1], self._step)
        y_steps = np.arange(0, image.shape[0], self._step)

        pad_width = x_steps[-1] + self._tile_size - image.shape[1]
        pad_height = y_steps[-1] + self._tile_size - image.shape[0]

        image = self._pad_image(image, pad_width, pad_height)

        tiles = list()
        for y in y_steps:
            for x in x_steps:
                tiles.append(image[y: y + self._tile_size, x: x + self._tile_size])

        return np.asarray(tiles)

    def _pad_image(self, image: np.ndarray, pad_width: int, pad_height: int):
        half_height = pad_height // 2
        half_width = pad_width // 2
        pads = [[half_height, pad_height - half_height], [half_width, pad_width - half_width]]
        if len(image.shape) == 3:
            pads.append([0, 0])

        image = np.pad(image, pad_width=pads, mode="constant", constant_values=self._fill_value)
        return image
