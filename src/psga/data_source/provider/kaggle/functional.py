import numpy as np


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def cut_tiles(image: np.ndarray, tile_size: int, border_value: int = 255) -> np.ndarray:
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





    return tiles
