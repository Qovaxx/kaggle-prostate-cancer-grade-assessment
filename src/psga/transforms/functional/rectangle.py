import cv2
import numpy as np

from .misc import crop_bbox
from ..entity import (
    BBox,
    CV2Rectangle
)

__all__ = ["crop_rectangle", "fast_crop_rectangle"]


def crop_rectangle(image: np.ndarray, rectangle: CV2Rectangle) -> np.ndarray:
    height, width = image.shape[:2]
    center_x, center_y = (width // 2, height // 2)
    transform = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=rectangle.angle, scale=1)
    cos = np.abs(transform[0, 0])
    sin = np.abs(transform[0, 1])
    enlarged_width = int((height * sin) + (width * cos))
    enlarged_height = int((height * cos) + (width * sin))
    transform[0, 2] += (enlarged_width / 2) - center_x
    transform[1, 2] += (enlarged_height / 2) - center_y

    if len(image.shape) == 3:
        flags = cv2.INTER_LANCZOS4
        border_value = (255, 255, 255)
    else:
        flags = cv2.INTER_NEAREST
        border_value = (0, 0, 0)

    image = cv2.warpAffine(src=image, M=transform, dsize=(enlarged_width, enlarged_height),
                           flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

    rotated_rectangle_center = np.dot(transform, np.array([rectangle.center_x, rectangle.center_y, 1]).T)
    rotated_rectangle = CV2Rectangle(center_x=rotated_rectangle_center[0],
                                     center_y=rotated_rectangle_center[1],
                                     width=rectangle.width,
                                     height=rectangle.height,
                                     angle=0)
    image = crop_bbox(image, bbox=rotated_rectangle.to_d4_bbox(), with_padding=True)
    return image


def fast_crop_rectangle(image: np.ndarray, rectangle: CV2Rectangle) -> np.ndarray:
    sub_images = list()
    bbox = cv2.boxPoints(box=rectangle.to_cv2_rect())
    round_side_correctly = lambda x, ids: np.floor(x / 2) if index in ids else np.ceil(x / 2)
    for index, corner in enumerate(bbox):
        sub_rectangle_center = np.mean([corner, [rectangle.center_x, rectangle.center_y]], axis=0)
        sub_rectangle = CV2Rectangle(center_x=sub_rectangle_center[0],
                                     center_y=sub_rectangle_center[1],
                                     width=round_side_correctly(int(rectangle.width), ids=[0, 1]),
                                     height=round_side_correctly(int(rectangle.height), ids=[1, 2]),
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
        sub_image = crop_bbox(image, sub_box)
        sub_images.append(crop_rectangle(sub_image, sub_rectangle))

    image = np.vstack([
        np.hstack([sub_images[1], sub_images[2]]),
        np.hstack([sub_images[0], sub_images[3]])
    ])
    return image
