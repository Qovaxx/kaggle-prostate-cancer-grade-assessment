from itertools import chain
from typing import (
    Optional,
    Tuple
)

import cv2
import rectpack
import numpy as np
from skimage import morphology

from . import functional as F
from .type import RECT_TYPE


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def minimize_background(image: np.ndarray, mask: Optional[np.ndarray] = None,
                        min_background_value: int = 255) -> Tuple[np.ndarray, Optional[np.ndarray], RECT_TYPE]:
    if mask is not None:
        assert image.shape[:2] == mask.shape

    gray = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
    _, roi = cv2.threshold(gray, thresh=min_background_value - 1, maxval=image.max(), type=cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(roi, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    rectangle = cv2.minAreaRect(points=np.concatenate(contours))
    cropped_image, cropped_mask = F.crop_rectangle(image, rectangle, mask)

    return cropped_image, cropped_mask, rectangle


def remove_gray_and_penmarks(image: np.ndarray, mask: Optional[np.ndarray] = None,
                             kernel_size: Tuple[int, int] = (5, 5), holes_objects_threshold_size: int = 1000,
                             max_gray_saturation: int = 5, red_left_shift: int = 50,
                             background_value: int = 255) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    if mask is not None:
        assert image.shape[:2] == mask.shape

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
    cleared_mask = tissue_mask * (1 - pen_marks_mask)

    cleared_mask = morphology.remove_small_holes(cleared_mask.astype(np.bool), area_threshold=holes_objects_threshold_size,
                                                 connectivity=1).astype(np.uint8)
    cleared_mask = morphology.remove_small_objects(cleared_mask.astype(np.bool), min_size=holes_objects_threshold_size,
                                                   connectivity=1).astype(np.uint8)
    cleared_mask = cv2.dilate(cleared_mask, kernel, iterations=2)
    cleared_mask = cv2.erode(cleared_mask, kernel, iterations=4)
    cleared_mask = cv2.dilate(cleared_mask, kernel, iterations=2)

    corrected_image = F.apply_mask_to_image(image, filter_mask=cleared_mask, add=background_value)
    corrected_mask = F.apply_mask_to_mask(mask, filter_mask=cleared_mask) if mask is not None else None

    return corrected_image, corrected_mask, cleared_mask


def convert_to_atlas(image: np.ndarray, tissue_mask: np.ndarray, mask: Optional[np.ndarray] = None,
                     step_size: int = 10):
    if mask is not None:
        assert image.shape[:2] == mask.shape == tissue_mask.shape
    contours, _ = cv2.findContours(tissue_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    pairs = list()

    for contour in contours:
        contour_mask = np.full(tissue_mask.shape, fill_value=0, dtype=np.uint8)
        cv2.drawContours(contour_mask, contours=[contour], contourIdx=0, color=1, thickness=-1)
        rectangle = cv2.minAreaRect(points=contour)
        contoured_image = F.apply_mask_to_image(image, filter_mask=contour_mask)
        contoured_mask = F.apply_mask_to_mask(mask, filter_mask=contour_mask) if mask is not None else None
        pairs.append(F.crop_rectangle(contoured_image, rectangle, contoured_mask))

    sides = list(chain(*[x[0].shape[:2] for x in pairs]))
    height = max(sides)
    width = min(sides)

    while True:
        packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline,
                                    pack_algo=rectpack.MaxRectsBlsf,
                                    sort_algo=rectpack.SORT_LSIDE,
                                    rotation=True)
        packer.add_bin(width=width, height=height)
        for index, pair in enumerate(pairs):
            packer.add_rect(width=pair[0].shape[1], height=pair[0].shape[0], rid=index)

        packer.pack()
        if len(pairs) == len(packer[0].rectangles):
            break
        else:
            width += step_size

    image_atlas = np.full(shape=(packer[0].height, packer[0].width, 3), fill_value=255, dtype=np.uint8)
    mask_atlas = None

    for rect in packer.rect_list():
        _, x, y, w, h, index = rect
        image_crop = pairs[index][0]
        if image_crop.shape[:2] != (h, w):
            image_crop = np.rot90(image_crop)
        image_atlas[y: y + h, x: x + w] = image_crop

    if mask is not None:
        mask_atlas = np.zeros(shape=(packer[0].height, packer[0].width), dtype=np.uint8)
        for rect in packer.rect_list():
            _, x, y, w, h, index = rect
            mask_crop = pairs[index][1]
            if mask_crop.shape != (h, w):
                mask_crop = np.rot90(mask_crop)
            mask_atlas[y: y + h, x: x + w] = mask_crop

    return image_atlas, mask_atlas
