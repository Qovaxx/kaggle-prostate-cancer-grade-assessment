from typing import (
    Optional,
    Tuple
)

import cv2
import numpy as np
from skimage import morphology

from . import functional as F

import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def remove_gray_and_penmarks(image: np.ndarray, mask: Optional[np.ndarray] = None,
                             kernel_size: Tuple[int, int] = (5, 5), holes_objects_threshold_size: int = 100,
                             max_gray_saturation: int = 5, red_left_shift: int = 50,
                             background_value: int = 255
                             ) -> Tuple[np.ndarray, np.ndarray]:
    if mask is None:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        tissue_mask = np.where(hsv[..., 1] >= max_gray_saturation, 1, 0).astype(np.uint8)

        # Mask pen marks pixels
        red_channel = image[..., 0]
        red_pixels = np.reshape(red_channel, (-1,))
        red_pixels = red_pixels[np.where(red_pixels < background_value)[0]]
        pen_marks_mask = (red_channel < np.median(red_pixels) - red_left_shift).astype(np.uint8)

        # Some tissue gets masked as well, thus needing to erode the mask to get rid of it.
        # Then some dilatation is applied to capture the "edges" of the "gradient-like"/non-uniform pen marks
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=kernel_size)
        pen_marks_mask = cv2.erode(pen_marks_mask, kernel, iterations=3)
        pen_marks_mask = cv2.dilate(pen_marks_mask, kernel, iterations=5)
        mask = tissue_mask * (1 - pen_marks_mask)

        mask = morphology.remove_small_holes(mask.astype(np.bool),
                                                     area_threshold=holes_objects_threshold_size,
                                                     connectivity=1).astype(np.uint8)
        mask = morphology.remove_small_objects(mask.astype(np.bool),
                                                       min_size=holes_objects_threshold_size,
                                                       connectivity=1).astype(np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=4)
        mask = cv2.dilate(mask, kernel, iterations=2)

    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, dsize=image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    if len(image.shape) == 3:
        image = F.apply_mask(image, mask=mask, add=background_value)
    else:
        image = F.apply_mask(mask, mask=mask)

    return image, mask
