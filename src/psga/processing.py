import gc
from typing import Tuple

import torch
import numpy as np

from .transforms.entity import Intermediates
from .transforms.atlas import convert_to_atlas
from .transforms.clear import remove_gray_and_penmarks
from .transforms.background import (
    crop_external_background,
    crop_inner_background
)

import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def _reduce_memory(image: np.ndarray) -> np.ndarray:
    image_gpu = torch.from_numpy(image).cuda()
    del image
    gc.collect()
    image = image_gpu.cpu().numpy()
    del image_gpu
    torch.cuda.empty_cache()
    return image


def dual_compose_preprocessing(large_image: np.ndarray, small_image: np.ndarray
                               )-> Tuple[np.ndarray, Intermediates]:
    scale = int(np.sqrt(large_image.size / small_image.size))
    small_image, intermediates = compose_preprocessing(small_image)
    del small_image
    gc.collect()

    intermediates.rescale(scale)
    large_image, intermediates = compose_preprocessing(large_image, intermediates, reduce_memory=True)

    return large_image, intermediates


def compose_preprocessing(image: np.ndarray, intermediates: Intermediates = Intermediates(),
                          reduce_memory: bool = False)-> Tuple[np.ndarray, Intermediates]:
    image, bbox = crop_external_background(image, bbox=intermediates.external_bbox)
    if reduce_memory:
        image = _reduce_memory(image)

    image, slice = crop_inner_background(image, slice=intermediates.inner_slice)
    if reduce_memory:
        image = _reduce_memory(image)

    image, tissue_objects = convert_to_atlas(image, tissue_objects=intermediates.tissue_objects)
    if reduce_memory:
        image = _reduce_memory(image)

    image, mask = remove_gray_and_penmarks(image, mask=intermediates.clear_mask)

    intermediates = Intermediates(bbox, slice, tissue_objects, mask)
    return image, intermediates
