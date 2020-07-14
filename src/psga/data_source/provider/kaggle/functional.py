from typing import (
    Optional,
    Tuple
)

import numpy as np


def cut_tiles(image: np.ndarray, tile_size: int, border_value: int = 255, filter_empty_threshold: float = 0.0,
              remove_empty_tiles: bool = False, return_normed_coords: bool = False
              ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    channels = image.shape[2]

    pad_h = (tile_size - image.shape[0] % tile_size) % tile_size
    pad_w = (tile_size - image.shape[1] % tile_size) % tile_size
    pad_width = [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]]
    image = np.pad(image, pad_width=pad_width, mode="constant", constant_values=border_value)

    n_tiles_h = image.shape[0] // tile_size
    n_tiles_w = image.shape[1] // tile_size
    tiles = image.reshape((n_tiles_h, tile_size, n_tiles_w, tile_size, channels))
    tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, channels)

    if remove_empty_tiles:
        empty_sum = tile_size * tile_size * channels * border_value * (1 - filter_empty_threshold)
        selected_indices = np.where(tiles.sum(axis=(1, 2, 3)) < empty_sum)[0]
        tiles = tiles[selected_indices]

    normed_coords = None
    if return_normed_coords:
        normed_coords = np.zeros((*tiles.shape[:3], 2), dtype=np.float32)
        for index in range(normed_coords.shape[0]):
            x = (index % n_tiles_w) * tile_size
            y = (index // n_tiles_w) * tile_size

            x_coords = np.tile(A=np.expand_dims(np.arange(x, x + tile_size), axis=0), reps=(tile_size, 1)) / image.shape[1]
            y_coords = np.tile(A=np.expand_dims(np.arange(y, y + tile_size), axis=1), reps=(1, tile_size)) / image.shape[0]
            normed_coords[index] = np.dstack([y_coords, x_coords])

    return tiles, normed_coords
