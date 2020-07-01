import gc
from typing import NoReturn

import numpy as np

from .transforms.entity import Intermediates
from .transforms.atlas import convert_to_atlas
from .transforms.clear import remove_gray_and_penmarks
from .transforms.background import (
    crop_external_background,
    crop_inner_background
)
from .utils.memory import reduce_numpy_memory


class ImagePreProcessor(object):

    def __init__(self, reduce_memory: bool = False) -> NoReturn:
        self._reduce_memory = reduce_memory
        self._intermediates = Intermediates()

    def rescale_intermediates(self, scale: int) -> NoReturn:
        self._intermediates.rescale(scale)

    def single(self, image: np.ndarray) -> np.ndarray:
        image, bbox = crop_external_background(image, bbox=self._intermediates.external_bbox)
        if self._reduce_memory:
            image = reduce_numpy_memory(image)

        image, slice = crop_inner_background(image, slice=self._intermediates.inner_slice)
        if self._reduce_memory:
            image = reduce_numpy_memory(image)

        image, rough_tissue_objects = convert_to_atlas(image, tissue_objects=self._intermediates.rough_tissue_objects)
        if self._reduce_memory:
            image = reduce_numpy_memory(image)

        image, mask = remove_gray_and_penmarks(image, mask=self._intermediates.clear_mask)
        image, precise_tissue_objects = convert_to_atlas(image, tissue_objects=self._intermediates.precise_tissue_objects,
                                                         not_background_mask=mask)

        self._intermediates = Intermediates(bbox, slice, rough_tissue_objects, mask, precise_tissue_objects)
        return image

    def dual(self, large_image: np.ndarray, small_image: np.ndarray) -> NoReturn:
        scale = int(np.sqrt(large_image.size / small_image.size))
        small_image = self.single(small_image)
        if self._reduce_memory:
            del small_image
            gc.collect()

        self._intermediates.rescale(scale)
        large_image = self.single(large_image)
        return large_image
