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
    image = image[slice.rows, :]
    return image[:, slice.columns]


def crop_rectangle(image: np.ndarray, rectangle: Rectangle, is_mask: bool = False) -> np.ndarray:
    height, width = image.shape[:2]
    center_x, center_y = (width // 2, height // 2)
    transform = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=rectangle.angle, scale=1)
    cos = np.abs(transform[0, 0])
    sin = np.abs(transform[0, 1])
    enlarged_width = int((height * sin) + (width * cos))
    enlarged_height = int((height * cos) + (width * sin))
    transform[0, 2] += (enlarged_width / 2) - center_x
    transform[1, 2] += (enlarged_height / 2) - center_y

    if is_mask:
        flags = cv2.INTER_NEAREST
        border_value = (0, 0, 0)
    else:
        flags = cv2.INTER_LANCZOS4
        border_value = (255, 255, 255)

    image = cv2.warpAffine(src=image, M=transform, dsize=(enlarged_width, enlarged_height),
                           flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


    rotated_rectangle_center = np.dot(transform, np.array([rectangle.center_x, rectangle.center_y, 1]).T)
    rotated_rectangle = Rectangle(center_x=rotated_rectangle_center[0],
                                  center_y=rotated_rectangle_center[1],
                                  width=rectangle.width,
                                  height=rectangle.height,
                                  angle=0)
    image = crop_bbox(image, bbox=rotated_rectangle.to_d4_bbox(), with_padding=True)
    return image
