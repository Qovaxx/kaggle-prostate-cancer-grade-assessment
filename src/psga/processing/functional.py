from typing import (
    Optional,
    Tuple
)

import cv2
import numpy as np

from .type import RECT_TYPE

import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def apply_mask_to_image(image: np.ndarray, filter_mask: np.ndarray, add: int = 255) -> np.ndarray:
    assert image.shape[:2] == filter_mask.shape
    assert np.array_equal(filter_mask, filter_mask.astype(bool))
    return image * filter_mask[..., np.newaxis] + add


def apply_mask_to_mask(mask: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
    assert mask.shape == filter_mask.shape
    assert np.array_equal(filter_mask, filter_mask.astype(bool))
    return mask * filter_mask


def crop_rectangle(image: np.ndarray, rectangle: RECT_TYPE, mask: Optional[np.ndarray] = None,
                   image_fill_value: int = 255, mask_fill_value: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    box = cv2.boxPoints(box=rectangle)
    width = int(rectangle[1][0])
    height = int(rectangle[1][1])
    reference_bbox = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype=np.float32)
    transformation_matrix = cv2.getPerspectiveTransform(src=box, dst=reference_bbox)

    crop = cv2.warpPerspective(image, M=transformation_matrix, dsize=(width, height),
                               flags=cv2.INTER_LANCZOS4, borderValue=tuple([image_fill_value] * 3))
    if mask is not None:
        mask = cv2.warpPerspective(mask, M=transformation_matrix, dsize=(width, height),
                                   flags=cv2.INTER_NEAREST, borderValue=mask_fill_value)

    return crop, mask











