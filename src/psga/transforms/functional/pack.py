from itertools import chain
from typing import (
    Tuple,
    List
)

import cv2
import rectpack
import numpy as np

from .misc import apply_mask
from .rectangle import fast_crop_rectangle
from ..entity import (
    TissueObjects,
    CV2Rectangle,
    RectPackRectangle
)

__all__ = ["create_bin", "pack_atlas"]


def create_bin(cv2_rectangles: List[CV2Rectangle], step_size: int = 10
               ) -> Tuple[Tuple[int, int], List[RectPackRectangle]]:
    sides = [(int(rect.width), int(rect.height)) for rect in cv2_rectangles]
    sides = list(chain(*sides))
    height = max(sides)
    width = min(sides)

    while True:
        packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline,
                                    pack_algo=rectpack.MaxRectsBlsf,
                                    sort_algo=rectpack.SORT_LSIDE,
                                    rotation=True)
        packer.add_bin(width=width, height=height)
        for index, rect in enumerate(cv2_rectangles):
            packer.add_rect(width=int(rect.width), height=int(rect.height), rid=index)

        packer.pack()
        if len(cv2_rectangles) == len(packer[0].rectangles):
            break
        else:
            width += step_size

    rectpack_rectangles = [RectPackRectangle(*rect) for rect in packer.rect_list()]

    return (height, width), rectpack_rectangles


def pack_atlas(image: np.ndarray, tissue_objects: TissueObjects) -> np.ndarray:
    fill_value = 255 if len(image.shape) == 3 else 0
    if image.shape[:2] != tissue_objects.mask.shape:
        tissue_objects.mask = cv2.resize(tissue_objects.mask, dsize=image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    crops = list()
    for index, rect in enumerate(tissue_objects.cv2_rectangles):
        mask = (tissue_objects.mask == index + 1).astype(np.uint8)
        contoured_image = apply_mask(image, mask=mask, add=fill_value)
        crops.append(fast_crop_rectangle(contoured_image, rect))

    shape = tissue_objects.bin_shape
    if len(image.shape) == 3:
        shape = [*shape, 3]

    atlas = np.full(shape=shape, fill_value=fill_value, dtype=np.uint8)
    for rect in tissue_objects.rectpack_rectangles:
        crop = crops[rect.index]
        if crop.shape[:2] != (rect.height, rect.width):
            crop = np.rot90(crop)
        atlas[rect.y: rect.y + rect.height, rect.x: rect.x + rect.width] = crop

    return atlas
