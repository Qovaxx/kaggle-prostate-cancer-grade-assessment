import gc
from typing import Tuple

import numpy as np

from .transforms.entity import Intermediates
from .transforms.atlas import convert_to_atlas
from .transforms.clear import remove_gray_and_penmarks
from .transforms.background import (
    crop_external_background,
    crop_inner_background
)
from .utils.memory import reduce_numpy_memory


def dual_compose_preprocessing(large_image: np.ndarray, small_image: np.ndarray
                               )-> Tuple[np.ndarray, Intermediates]:
    scale = int(np.sqrt(large_image.size / small_image.size))
    small_image, intermediates = compose_preprocessing(small_image)
    del small_image
    gc.collect()

    print("RESCALE")
    intermediates.rescale(scale)
    print("GO2")
    large_image, intermediates = compose_preprocessing(large_image, intermediates, reduce_memory=True)

    return large_image, intermediates


def compose_preprocessing(image: np.ndarray, intermediates: Intermediates = Intermediates(),
                          reduce_memory: bool = False)-> Tuple[np.ndarray, Intermediates]:
    print("crop_external_background")
    image, bbox = crop_external_background(image, bbox=intermediates.external_bbox)
    if reduce_memory:
        image = reduce_numpy_memory(image)

    print("crop_inner_background")
    image, slice = crop_inner_background(image, slice=intermediates.inner_slice)
    if reduce_memory:
        image = reduce_numpy_memory(image)

    print("convert_to_atlas")
    image, rough_tissue_objects = convert_to_atlas(image, tissue_objects=intermediates.rough_tissue_objects)
    if reduce_memory:
        image = reduce_numpy_memory(image)

    print("remove_gray_and_penmarks")
    image, mask = remove_gray_and_penmarks(image, mask=intermediates.clear_mask)
    print("prec convert_to_atlas")
    image, precise_tissue_objects = convert_to_atlas(image, tissue_objects=intermediates.precise_tissue_objects,
                                                     not_background_mask=mask)

    intermediates = Intermediates(bbox, slice, rough_tissue_objects, mask, precise_tissue_objects)
    return image, intermediates
