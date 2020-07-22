from typing import (
    Optional,
    Tuple
)

import cv2
import numpy as np

from . import functional as F
from .entity import (
    BBox,
    Slice2D,
    CV2Rectangle
)


def crop_external_background(image: np.ndarray, bbox: Optional[BBox] = None,
                             background_value: int = 255) -> Tuple[np.ndarray, BBox]:
    if bbox is None:
        height_indices = (image.min(axis=(1, 2)) < background_value).nonzero()[0]
        width_indices = (image.min(axis=(0, 2)) < background_value).nonzero()[0]
        bbox = BBox(x=width_indices[0],
                    y=height_indices[0],
                    width=width_indices[-1] - width_indices[0],
                    height=height_indices[-1] - height_indices[0])

    image = F.crop_bbox(image, bbox)
    return image, bbox


def crop_inner_background(image: np.ndarray, slice: Optional[Slice2D] = None,
                          background_value: int = 255) -> Tuple[np.ndarray, Slice2D]:
    if slice is None:
        background_mask = (image == background_value)
        to_boolist = lambda axis: [x.all() for x in ~np.all(background_mask, axis=axis)]
        slice = Slice2D(rows=to_boolist(axis=1), columns=to_boolist(axis=0))

    image = F.crop_slice(image, slice)
    return image, slice


def crop_minimum_roi(image: np.ndarray, rectangle: Optional[CV2Rectangle] = None,
                     background_value: int = 255) -> Tuple[np.ndarray, CV2Rectangle]:
    if rectangle is None:
        gray = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
        _, roi = cv2.threshold(gray, thresh=background_value - 1, maxval=image.max(), type=cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(roi, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        rectangle = cv2.minAreaRect(points=np.concatenate(contours))
        rectangle = CV2Rectangle(center_x=rectangle[0][0],
                                 center_y=rectangle[0][1],
                                 width=rectangle[1][0],
                                 height=rectangle[1][1],
                                 angle=rectangle[2])

    image = F.fast_crop_rectangle(image, rectangle)
    return image, rectangle
