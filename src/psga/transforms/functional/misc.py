import cv2
import numpy as np

from ..entity import (
    BBox,
    Slice2D
)

__all__ = ["apply_mask", "crop_bbox", "crop_slice"]


def apply_mask(image: np.ndarray, mask: np.ndarray, add: int = 0) -> np.ndarray:
    return cv2.bitwise_and(src1=image, src2=image, mask=mask) + add


def crop_bbox(image: np.ndarray, bbox: BBox, with_padding: bool = False) -> np.ndarray:
    x_max = bbox.x + bbox.width
    y_max = bbox.y + bbox.height
    pads = (bbox.x, bbox.y, image.shape[1]-x_max, image.shape[0]-y_max)
    correct_pads = len(list(filter(lambda x: x>=0, pads)))

    if with_padding is False or correct_pads == len(pads):
        return image[bbox.y: bbox.y + bbox.height, bbox.x: bbox.x + bbox.width]

    else:
        pads = list(map(lambda x: 0 if x > 0 else abs(x), pads))
        pad_width = [(pads[1], pads[3]), (pads[0], pads[2])]
        constant_values = 0
        if len(image.shape) == 3:
            pad_width.append((0, 0))
            constant_values = 255

        padded = np.pad(image, pad_width=pad_width, mode="constant", constant_values=constant_values)
        return padded[
               pads[1] + bbox.y: pads[1] + bbox.y + bbox.height,
               pads[0] + bbox.x: pads[0] + bbox.x + bbox.width
               ]


def crop_slice(image: np.ndarray, slice: Slice2D) -> np.ndarray:
    image = np.take(image, indices=np.asarray(slice.rows).nonzero()[0], axis=0)
    image = np.take(image, indices=np.asarray(slice.columns).nonzero()[0], axis=1)
    return image
