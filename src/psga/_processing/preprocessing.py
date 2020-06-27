from typing import (
    Optional,
    Tuple,
    List
)

import cv2
import numpy as np
from skimage import morphology

from . import functional as F
from .types import (
    Contour,
    IMAGE_TYPE,
    MASK_TYPE,
    CV2_RECT_TYPE,
    SLICE_TYPE
)

from time import time
import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def reduce_external_background(image: IMAGE_TYPE, mask: MASK_TYPE = None,
                               bbox: Optional[np.ndarray] = None, background_value: int = 255
                               ) -> Tuple[IMAGE_TYPE, MASK_TYPE, np.ndarray]:
    F.validate_shape(image, mask)

    if bbox is None:
        height_indices = (image.min(axis=(1, 2)) < background_value).nonzero()[0]
        width_indices = (image.min(axis=(0, 2)) < background_value).nonzero()[0]
        bbox = np.array([[width_indices[0], height_indices[0]], [width_indices[-1], height_indices[-1]]])

    reduced_image = F.crop_bbox(image, bbox)
    reduced_mask = None
    if mask is not None:
        reduced_mask = F.crop_bbox(mask, bbox)
    return reduced_image, reduced_mask, bbox


def reduce_inner_background(image: IMAGE_TYPE, mask: MASK_TYPE = None,
                            row_slice: Optional[SLICE_TYPE] = None, column_slice: Optional[SLICE_TYPE] = None,
                            background_value: int = 255,
                            ) -> Tuple[IMAGE_TYPE, MASK_TYPE, SLICE_TYPE, SLICE_TYPE]:
    F.validate_shape(image, mask)

    if row_slice is None and column_slice is None:
        background_mask = (image == background_value)
        to_bool_list = lambda axis: [x.all() for x in ~np.all(background_mask, axis=axis)]
        row_slice = to_bool_list(axis=1)
        column_slice = to_bool_list(axis=0)

    reduced_image = F.crop_slices(image, row_slice, column_slice)
    reduced_mask = None
    if mask is not None:
        reduced_mask = F.crop_slices(mask, row_slice, column_slice)
    return reduced_image, reduced_mask, row_slice, column_slice


def crop_roi(image: IMAGE_TYPE, mask: MASK_TYPE = None,
             rectangle: Optional[CV2_RECT_TYPE] = None,
             background_value: int = 255
             ) -> Tuple[IMAGE_TYPE, MASK_TYPE, CV2_RECT_TYPE]:
    F.validate_shape(image, mask)

    if rectangle is None:
        gray = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
        _, roi = cv2.threshold(gray, thresh=background_value - 1, maxval=image.max(), type=cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(roi, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        rectangle = cv2.minAreaRect(points=np.concatenate(contours))

    cropped_image, cropped_mask = F.crop_rectangle(image, mask=mask, rectangle=rectangle)
    return cropped_image, cropped_mask, rectangle








# TODO: REFACTOR

def remove_gray_and_penmarks(image: np.ndarray, mask: Optional[np.ndarray] = None,
                             kernel_size: Tuple[int, int] = (5, 5), holes_objects_threshold_size: int = 100,
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

    cleared_mask = morphology.remove_small_holes(cleared_mask.astype(np.bool),
                                                 area_threshold=holes_objects_threshold_size,
                                                 connectivity=1).astype(np.uint8)
    cleared_mask = morphology.remove_small_objects(cleared_mask.astype(np.bool),
                                                   min_size=holes_objects_threshold_size,
                                                   connectivity=1).astype(np.uint8)
    cleared_mask = cv2.dilate(cleared_mask, kernel, iterations=2)
    cleared_mask = cv2.erode(cleared_mask, kernel, iterations=4)
    cleared_mask = cv2.dilate(cleared_mask, kernel, iterations=2)

    corrected_image = F.apply_mask(image, mask=cleared_mask, add=background_value)
    corrected_mask = F.apply_mask(mask, mask=cleared_mask) if mask is not None else None

    return corrected_image, corrected_mask, cleared_mask


def convert_to_atlas(image: np.ndarray, tissue_mask: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], List[Contour]]:
    if mask is not None:
        assert image.shape[:2] == mask.shape == tissue_mask.shape
    found, _ = cv2.findContours(tissue_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = list()

    for contour in found:
        contour_mask = np.full(tissue_mask.shape, fill_value=0, dtype=np.uint8)
        cv2.drawContours(contour_mask, contours=[contour], contourIdx=0, color=1, thickness=-1)
        rectangle = cv2.minAreaRect(points=contour)
        contours.append(Contour(mask=contour_mask, rectangle=rectangle))

    rectangles, atlas_shape = F.pack_bin(contours)
    image_atlas, mask_atlas = F.pack_atlas(image, mask=mask, atlas_shape=atlas_shape,
                                           contours=contours, rectangles=rectangles)

    return image_atlas, mask_atlas, contours


def compose_preprocessing(master_image: IMAGE_TYPE, minion_image: IMAGE_TYPE,
                          master_mask: MASK_TYPE = None, minion_mask: MASK_TYPE = None,
                          roi_acceptance_threshold: float = 1.5,
                          ) -> Tuple[IMAGE_TYPE, MASK_TYPE]:

    scale = int(master_image.shape[0] / minion_image.shape[0])

    # Reduce empty white external background
    minion_image_ex_reduced, minion_mask_ex_reduced, bbox = \
        reduce_external_background(image=minion_image, mask=minion_mask)
    # master_image_ex_reduced, master_mask_ex_reduced, _ = \
    #     reduce_external_background(image=master_image, mask=master_mask, bbox=bbox * scale)

    # Reduce empty white inner background
    minion_image_in_reduced, minion_mask_in_reduced, row_slice, column_slice = \
        reduce_inner_background(image=minion_image_ex_reduced, mask=minion_mask_ex_reduced)
    # master_image_in_reduced, master_mask_in_reduced, _, _ = \
    #     reduce_inner_background(image=master_image_ex_reduced, mask=master_mask_ex_reduced,
    #                             row_slice=F.scale_slice(row_slice, master_image_ex_reduced.shape[0]),
    #                             column_slice=F.scale_slice(column_slice, master_image_ex_reduced.shape[1]))


























    # # Crop the smallest possible region of interest
    # minion_image_roi, minion_mask_roi, rectangle = crop_roi(image=minion_image_in_reduced, mask=minion_mask_in_reduced)
    # if minion_image_in_reduced.size / minion_image_roi.size > roi_acceptance_threshold:
    #     print(f"ACCEPT {minion_image_in_reduced.size / minion_image_roi.size}")
    #     master_image_roi, master_mask_roi = F.crop_rectangle(image=master_image_in_reduced, mask=master_mask_in_reduced,
    #                                                          rectangle=F.scale_rectangle(rectangle, scale))
    # else:
    #     print(f"FAIL {minion_image_in_reduced.size / minion_image_roi.size}")
    #     minion_image_roi = minion_image_in_reduced
    #     minion_mask_roi = minion_mask_in_reduced
    #     master_image_roi = master_image_in_reduced
    #     master_mask_roi = master_mask_in_reduced
    #
    #
    #
    # # # Crop minimum roi
    # # minion_image_roi, minion_mask_roi, rectangle_roi = crop_roi(image=minion_image_ex_reduced, mask=minion_mask_ex_reduced)
    # # scaled_rectangle_roi = F.scale_rectangle(rectangle_roi, scale)
    # # master_image_roi, master_mask_roi = F.crop_rectangle(image=master_image_ex_reduced, mask=master_mask_ex_reduced,
    # #                                                      rectangle=scaled_rectangle_roi)



    # # Clear from gray artifacts and pen marks
    # minion_image_clear, minion_mask_clear, tissue_mask = \
    #     remove_gray_and_penmarks(image=minion_image_roi,
    #                              mask=minion_mask_roi,
    #                              kernel_size=kernel_size,
    #                              holes_objects_threshold_size=holes_objects_threshold_size)
    # scaled_tissue_mask = cv2.resize(tissue_mask, dsize=master_image_roi.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    # master_image_clear = F.apply_mask(master_image_roi, mask=scaled_tissue_mask, add=background_value)
    # master_mask_clear = F.apply_mask(master_mask_roi, mask=scaled_tissue_mask) if master_mask_roi is not None else None
    #
    # minion_image_atlas, minion_mask_atlas, contours = convert_to_atlas(image=minion_image_clear,
    #                                                                    mask=minion_mask_clear,
    #                                                                    tissue_mask=tissue_mask)
    # scaled_contours = list()
    # for contour in contours:
    #     scaled_contour_mask = cv2.resize(contour.mask, dsize=master_image_clear.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    #     scaled_contours.append(Contour(mask=scaled_contour_mask, rectangle=F.scale_rectangle(contour.rectangle, scale)))
    #
    # rectangles, atlas_shape = F.pack_bin(scaled_contours)
    # master_image_atlas, master_mask_atlas = F.pack_atlas(image=master_image_clear,
    #                                                      mask=master_mask_clear,
    #                                                      atlas_shape=atlas_shape,
    #                                                      contours=scaled_contours,
    #                                                      rectangles=rectangles)
    #
    # return master_image_atlas, master_mask_atlas
