from itertools import chain
from typing import (
    Tuple,
    List
)

import rectpack
import numpy as np

from .misc import apply_mask
from .rectangle import fast_crop_rectangle
from ..entity import TissueObjects

__all__ = ["create_bin", "pack_atlas"]

RECTPACK_RECT_TYPE = Tuple[int, int, int, int, int, int]


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def create_bin(tissue_objects: TissueObjects, step_size: int = 10
               ) -> Tuple[Tuple[int, int], List[RECTPACK_RECT_TYPE]]:
    sides = [(int(rect.width), int(rect.height)) for rect in tissue_objects.rectangles]
    sides = list(chain(*sides))
    height = max(sides)
    width = min(sides)

    while True:
        packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline,
                                    pack_algo=rectpack.MaxRectsBlsf,
                                    sort_algo=rectpack.SORT_LSIDE,
                                    rotation=True)
        packer.add_bin(width=width, height=height)
        for index, rect in enumerate(tissue_objects.rectangles):
            packer.add_rect(width=int(rect.width), height=int(rect.height), rid=index)

        packer.pack()
        if len(tissue_objects.rectangles) == len(packer[0].rectangles):
            break
        else:
            width += step_size

    return (height, width), packer.rect_list()


def pack_atlas(image: np.ndarray, tissue_objects: TissueObjects) -> np.ndarray:
    fill_value = 255 if len(image.shape) == 3 else 0
    crops = list()
    for index, rectangle in enumerate(tissue_objects.rectangles):
        mask = (tissue_objects.mask == index + 1).astype(np.uint8)
        contoured_image = apply_mask(image, mask=mask, add=fill_value)
        crops.append(fast_crop_rectangle(contoured_image, rectangle))

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

    return atlas
