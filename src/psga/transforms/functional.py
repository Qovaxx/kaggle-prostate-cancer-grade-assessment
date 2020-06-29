import cv2
import numpy as np

from .entity import (
    BBox,
    Slice2D,
    Rectangle
)


def crop_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
    return image[bbox.y: bbox.y + bbox.height, bbox.x: bbox.x + bbox.width]


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
    image = crop_bbox(image, bbox=rotated_rectangle.to_d4_bbox())
    return image
