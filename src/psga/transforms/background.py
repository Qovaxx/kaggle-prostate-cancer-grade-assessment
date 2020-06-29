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
    Rectangle
)
from ..settings import CV2_MAX_IMAGE_SIZE

from time import time
import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()



def crop_external_background(image: np.ndarray, bbox: Optional[BBox] = None,
                             background_value: int = 255) -> Tuple[np.ndarray, BBox]:
    if bbox is None:
        height_indices = (image.min(axis=(1, 2)) < background_value).nonzero()[0]
        width_indices = (image.min(axis=(0, 2)) < background_value).nonzero()[0]
        bbox = BBox(x=width_indices[0],
                    y=height_indices[0],
                    width=width_indices[-1] - width_indices[0],
                    height=height_indices[-1] - height_indices[0])

    print(f"crop bbox begin {image.shape}")
    image = F.crop_bbox(image, bbox)
    print(f"crop cropped {image.shape}")
    return image, bbox


def crop_inner_background(image: np.ndarray, slice: Optional[Slice2D] = None,
                          background_value: int = 255) -> Tuple[np.ndarray, Slice2D]:
    if slice is None:
        background_mask = (image == background_value)
        to_boolist = lambda axis: [x.all() for x in ~np.all(background_mask, axis=axis)]
        slice = Slice2D(rows=to_boolist(axis=1), columns=to_boolist(axis=0))

    print(f"crop slice begin {image.shape}")
    image = F.crop_slice(image, slice)
    print(f"slice cropped {image.shape}")
    return image, slice


def crop_minimum_roi(image: np.ndarray, rectangle: Optional[Rectangle] = None,
                     is_mask: bool = False, background_value: int = 255) -> Tuple[np.ndarray, Rectangle]:
    if rectangle is None:
        gray = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
        _, roi = cv2.threshold(gray, thresh=background_value - 1, maxval=image.max(), type=cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(roi, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        rectangle = cv2.minAreaRect(points=np.concatenate(contours))
        rectangle = Rectangle(center_x=rectangle[0][0],
                              center_y=rectangle[0][1],
                              width=rectangle[1][0],
                              height=rectangle[1][1],
                              angle=rectangle[2])

    show(image)
    print(f"crop rectangle begin {image.shape}")

    start = time()
    sub_images = list()
    bbox = cv2.boxPoints(box=rectangle.to_cv2_rect())

    for corner in bbox:
        sub_rectangle_center = np.mean([corner, [rectangle.center_x, rectangle.center_y]], axis=0)
        sub_rectangle = Rectangle(center_x=sub_rectangle_center[0],
                                  center_y=sub_rectangle_center[1],
                                  width=int(rectangle.width / 2),
                                  height=int(rectangle.height / 2),
                                  angle=rectangle.angle)

        sub_box = np.int0(cv2.boxPoints(box=sub_rectangle.to_cv2_rect()))
        corners_x, corners_y = sub_box[:, 0], sub_box[:, 1]
        min_x, min_y, max_x, max_y = min(corners_x), min(corners_y), max(corners_x), max(corners_y)
        correct_min_x, correct_min_y = max(0, min_x), max(0, min_y)
        sub_box = BBox(x=correct_min_x,
                       y=correct_min_y,
                       width=min(image.shape[1], max_x - correct_min_x),
                       height=min(image.shape[0], max_y - correct_min_y))
        sub_rectangle.center_x -= correct_min_x
        sub_rectangle.center_y -= correct_min_y
        sub_image = F.crop_bbox(image, sub_box)
        sub_images.append(F.crop_rectangle(sub_image, sub_rectangle, is_mask=is_mask))

    image1 = np.vstack([
        np.hstack([sub_images[1], sub_images[2]]),
        np.hstack([sub_images[0], sub_images[3]])
    ])
    print(time() - start)

    start = time()
    image2 = F.crop_rectangle(image, rectangle, is_mask=is_mask)
    print(time() - start)

    print(f"rectangle cropped {image.shape}")
    show(image1)
    show(image2)
    return image, rectangle
