from typing import NoReturn

import numpy as np
from pyvips.enums import ForeignTiffCompression

from .vips import from_numpy


def save_tiff_image(image: np.ndarray, path: str, quality: int = 90, tile_size: int = 512) -> NoReturn:
    from_numpy(image).write_to_file(path, bigtiff=True, compression=ForeignTiffCompression.JPEG, Q=quality,
                                    tile=True, tile_width=tile_size, tile_height=tile_size, rgbjpeg=True)


def save_tiff_mask(mask: np.ndarray, path: str, tile_size: int = 512) -> NoReturn:
    from_numpy(mask).write_to_file(path, bigtiff=True, compression=ForeignTiffCompression.LZW,
                                   tile=True, tile_width=tile_size, tile_height=tile_size, rgbjpeg=False)

