from dataclasses import dataclass
from typing import (
    Optional,
    Tuple,
    NoReturn
)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_pen_marks(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), max_tissue_value: int = 220,
                     red_left_shift: int = 50, background_value: int = 255) -> Tuple[np.ndarray, np.ndarray]:
    corrected = image.copy()
    gray = cv2.cvtColor(corrected, code=cv2.COLOR_RGB2GRAY)
    non_background_mask = np.where(gray < max_tissue_value, 1, 0).astype(np.uint8)
    plt.imshow(non_background_mask)
    plt.show()

    # Mask pen marks pixels
    red_pixels = np.reshape(corrected[:, :, 0], (-1,))
    red_pixels = red_pixels[np.where(red_pixels < background_value)[0]]
    pen_marks_mask = (corrected[:, :, 0] < np.median(red_pixels) - red_left_shift).astype(np.uint8)

    # When computing the pen mark mask, some tissue gets masked as well,
    # thus needing to erode the mask to get rid of it. Then some dilatation is
    # applied to capture the "edges" of the "gradient-like"/non-uniform pen marks
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=kernel_size) # kernel size selected for level 1
    pen_marks_mask = cv2.erode(pen_marks_mask, kernel, iterations=3)
    pen_marks_mask = cv2.dilate(pen_marks_mask, kernel, iterations=5)

    pen_marks_mask = 1 - pen_marks_mask
    mask = non_background_mask * pen_marks_mask

    # Fill some gaps/holes in the tissue
    mask = cv2.morphologyEx(mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, 1, -1)

    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=2)
    corrected = corrected * mask[..., np.newaxis] + background_value
    return corrected, mask


@dataclass
class BBox(object):
    x1: int
    y1: int
    x2: int
    y2: int
    shape: Tuple[int, int]

    def rescale(self, target_shape: Tuple[int, int]):
        x_scale = target_shape[0] / self.shape[0]
        y_scale = target_shape[1] / self.shape[1]
        return BBox(x1=int(self.x1 * x_scale),
                    y1=int(self.y1 * y_scale),
                    x2=int(self.x2 * x_scale),
                    y2=int(self.y2 * y_scale),
                    shape=target_shape)


class NoneWhiteROICropper(object):

    def __init__(self, min_background_value: int = 255) -> NoReturn:
        self._min_background_value = min_background_value
        self._bbox: Optional[BBox] = None

    @property
    def bbox(self) -> Optional[BBox]:
        return self._bbox

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                 bbox: Optional[BBox] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if mask is not None:
            assert image.shape[:2] == mask.shape[:2]

        if bbox is None:
            height_indices = (image.min(axis=(1, 2)) < self._min_background_value).nonzero()[0]
            width_indices = (image.min(axis=(0, 2)) < self._min_background_value).nonzero()[0]
            bbox = BBox(width_indices[0], width_indices[-1], height_indices[0], height_indices[-1], image.shape[:2])
        self._bbox = bbox
        crop_it = lambda x: x[bbox.x2: bbox.y2, bbox.x1: bbox.y1]

        image = crop_it(image)
        if mask is not None:
            mask = crop_it(mask)
        return image, mask
