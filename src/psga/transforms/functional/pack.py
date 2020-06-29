from itertools import chain
from typing import (
    Tuple,
    List
)

import rectpack
import numpy as np

from .misc import apply_mask
from .rectangle import fast_crop_rectangle, crop_rectangle
from ..entity import TissueObject

__all__ = ["create_bin", "pack_atlas"]

RECTPACK_RECT_TYPE = Tuple[int, int, int, int, int, int]


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def create_bin(tissue_objects: List[TissueObject], step_size: int = 10
               ) -> Tuple[Tuple[int, int], List[RECTPACK_RECT_TYPE]]:
    sides = [(int(obj.rectangle.width), int(obj.rectangle.height)) for obj in tissue_objects]
    sides = list(chain(*sides))
    height = max(sides)
    width = min(sides)

    while True:
        packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline,
                                    pack_algo=rectpack.MaxRectsBlsf,
                                    sort_algo=rectpack.SORT_LSIDE,
                                    rotation=True)
        packer.add_bin(width=width, height=height)
        for index, obj in enumerate(tissue_objects):
            packer.add_rect(width=int(obj.rectangle.width), height=int(obj.rectangle.height), rid=index)

        packer.pack()
        if len(tissue_objects) == len(packer[0].rectangles):
            break
        else:
            width += step_size

    return (height, width), packer.rect_list()


def pack_atlas(image: np.ndarray, tissue_objects: List[TissueObject]) -> np.ndarray:
    fill_value = 255 if len(image.shape) == 3 else 0
    crops = list()
    for obj in tissue_objects:
        contoured_image = apply_mask(image, mask=obj.mask.astype(np.uint8), add=fill_value)
        crops.append(fast_crop_rectangle(contoured_image, obj.rectangle))

    shape, rectangles = create_bin(tissue_objects)
    if len(image.shape) == 3:
        shape = [*shape, 3]

    atlas = np.full(shape=shape, fill_value=fill_value, dtype=np.uint8)
    for rectangle in rectangles:
        _, x, y, w, h, index = rectangle
        crop = crops[index]
        if crop.shape[:2] != (h, w):
            crop = np.rot90(crop)
        atlas[y: y + h, x: x + w] = crop

    return image
