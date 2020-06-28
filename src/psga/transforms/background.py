from typing import (
    Optional,
    Tuple
)

import cv2
import numpy as np

from .entity import (
    BBox,
    Slice2D,
    Rectangle
)

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

    image = image[bbox.y: bbox.y + bbox.height, bbox.x: bbox.x + bbox.width]
    return image, bbox


def crop_inner_background(image: np.ndarray, slice: Optional[Slice2D] = None,
                          background_value: int = 255) -> Tuple[np.ndarray, Slice2D]:
    if slice is None:
        background_mask = (image == background_value)
        to_boolist = lambda axis: [x.all() for x in ~np.all(background_mask, axis=axis)]
        slice = Slice2D(rows=to_boolist(axis=1), columns=to_boolist(axis=0))

    image = image[slice.rows, :]
    image = image[:, slice.columns]
    return image, slice


def crop_minimum_roi(image: np.ndarray, rectangle: Optional[Rectangle] = None,
                     background_value: int = 255) -> Tuple[np.ndarray, Rectangle]:
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

    z = image.copy()
    bbox = cv2.boxPoints(box=rectangle.to_cv2())
    cv2.drawContours(z, [np.int0(bbox)], 0, (0, 255, 255), 4)
    cv2.circle(z, (int(rectangle.center_x), int(rectangle.center_y)), radius=30, color=(255, 0, 0), thickness=-1)
    show(z)

    height, width = image.shape[:2]
    center_x, center_y = (width // 2, height // 2)
    transform = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=rectangle.angle, scale=1)
    cos = np.abs(transform[0, 0])
    sin = np.abs(transform[0, 1])
    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))
    transform[0, 2] += (nW / 2) - center_x
    transform[1, 2] += (nH / 2) - center_y

    warped = cv2.warpAffine(src=image, M=transform, dsize=(nW, nH))

    calculated = np.dot(transform, np.array([rectangle.center_x, rectangle.center_y, 1]).T)
    rectangle2 = Rectangle(
        center_x=calculated[0],
        center_y=calculated[1],
        width=rectangle.width,
        height=rectangle.height,
        angle=0
    )
    z = warped.copy()
    bbox = cv2.boxPoints(box=rectangle2.to_cv2())
    cv2.drawContours(z, [np.int0(bbox)], 0, (0, 255, 255), 4)
    cv2.circle(z, (int(rectangle2.center_x), int(rectangle2.center_y)), radius=30, color=(255, 0, 0), thickness=-1)
    show(z)

    a = 4




    return image, rectangle
