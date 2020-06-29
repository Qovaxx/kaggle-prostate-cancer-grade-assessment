import gc
from typing import Tuple

import torch
import numpy as np

from .transforms.entity import Intermediates
from .transforms.background import (
    crop_external_background,
    crop_inner_background,
    crop_minimum_roi
)

import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def _reduce_memory(image: np.ndarray):
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
    print(f"DONE {large_image.shape}")

    return large_image, intermediates


def compose_preprocessing(image: np.ndarray, intermediates: Intermediates = Intermediates(),
                          is_mask: bool = False,
                          reduce_memory: bool = False)-> Tuple[np.ndarray, Intermediates]:
    image, bbox = crop_external_background(image, bbox=intermediates.external_bbox)
    if reduce_memory:
        image = _reduce_memory(image)

    image, slice = crop_inner_background(image, slice=intermediates.inner_slice)
    if reduce_memory:
        image = _reduce_memory(image)

    image, rectangle = crop_minimum_roi(image, rectangle=intermediates.roi_rectangle, is_mask=is_mask)
    if reduce_memory:
        image = _reduce_memory(image)

    # crop_inner_background

    intermediates = Intermediates(bbox, slice, rectangle)
    return image, intermediates
