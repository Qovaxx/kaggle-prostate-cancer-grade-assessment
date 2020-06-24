from skimage import morphology
from typing import (
    Optional,
    Tuple
)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def crop_rectangle(image: np.ndarray, mask: Optional[np.ndarray], rectangle: Tuple[Tuple, Tuple, float],
                   border_value: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    box = cv2.boxPoints(box=rectangle)
    width = int(rectangle[1][0])
    height = int(rectangle[1][1])
    reference_bbox = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype=np.float32)
    transformation_matrix = cv2.getPerspectiveTransform(src=box, dst=reference_bbox)

    crop = cv2.warpPerspective(image, M=transformation_matrix, dsize=(width, height),
                               flags=cv2.INTER_LANCZOS4, borderValue=tuple([border_value] * 3))
    if mask is not None:
        mask = cv2.warpPerspective(mask, M=transformation_matrix, dsize=(width, height),
                                   flags=cv2.INTER_NEAREST, borderValue=0)

    return crop, mask


def crop_min_roi(image: np.ndarray, mask: Optional[np.ndarray] = None, min_background_value: int = 255
                 ) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[Tuple, Tuple, float]]:
    if mask is not None:
        assert image.shape[:2] == mask.shape

    gray = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
    _, roi = cv2.threshold(gray, thresh=min_background_value - 1, maxval=image.max(), type=cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(roi, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    rectangle = cv2.minAreaRect(points=np.concatenate(contours))
    cropped_image, cropped_mask = crop_rectangle(image, mask, rectangle, border_value=min_background_value)

    return cropped_image, cropped_mask, rectangle


def remove_pen_marks(image: np.ndarray, mask: Optional[np.ndarray], kernel_size: Tuple[int, int],
                     max_tissue_value: int = 220, red_left_shift: int = 50,
                     background_value: int = 255) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    if mask is not None:
        assert image.shape[:2] == mask.shape

    gray = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
    non_gray_mask = np.where(gray < max_tissue_value, 1, 0).astype(np.uint8)

    # Mask pen marks pixels
    red_pixels = np.reshape(image[:, :, 0], (-1,))
    red_pixels = red_pixels[np.where(red_pixels < background_value)[0]]
    pen_marks_mask = (image[:, :, 0] < np.median(red_pixels) - red_left_shift).astype(np.uint8)

    # Some tissue gets masked as well, thus needing to erode the mask to get rid of it.
    # Then some dilatation is applied to capture the "edges" of the "gradient-like"/non-uniform pen marks
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=kernel_size)
    pen_marks_mask = cv2.erode(pen_marks_mask, kernel, iterations=3)
    pen_marks_mask = cv2.dilate(pen_marks_mask, kernel, iterations=5)
    tissue_mask = non_gray_mask * (1 - pen_marks_mask)

    morphology.remove_small_holes(tissue_mask.astype(np.bool), area_threshold=1000, connectivity=1, in_place=True)
    tissue_mask = cv2.dilate(tissue_mask, kernel, iterations=2)
    tissue_mask = cv2.erode(tissue_mask, kernel, iterations=4)
    tissue_mask = cv2.dilate(tissue_mask, kernel, iterations=2)

    corrected_image = image * tissue_mask[..., np.newaxis] + background_value
    corrected_mask = mask * tissue_mask if mask is not None else None

    return corrected_image, corrected_mask, tissue_mask


def convert_to_atlas(image: np.ndarray, mask: Optional[np.ndarray])

