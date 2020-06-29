from typing import (
    List,
    Tuple,
    Optional
)

import cv2
import numpy as np

from . import functional as F
from .entity import (
    TissueObjects,
    Rectangle
)


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def convert_to_atlas(image: np.ndarray, tissue_objects: Optional[TissueObjects] = None,
                     min_contour_area: int = 200,
                     background_value: int = 255) -> Tuple[np.ndarray, TissueObjects]:
    if tissue_objects is None:
        gray = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
        _, not_background_mask = cv2.threshold(gray, thresh=background_value - 1, maxval=image.max(),
                                               type=cv2.THRESH_BINARY_INV)
        found_contours, _ = cv2.findContours(not_background_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        rectangles = list()
        mask = np.full(image.shape[:2], fill_value=0, dtype=np.uint8)

        for contour in found_contours:
            if cv2.contourArea(contour) >= min_contour_area:
                color = len(rectangles) + 1
                cv2.drawContours(mask, contours=[contour], contourIdx=0, color=color, thickness=-1)
                rectangle = cv2.minAreaRect(points=contour)
                rectangles.append(Rectangle(center_x=rectangle[0][0],
                                            center_y=rectangle[0][1],
                                            width=rectangle[1][0],
                                            height=rectangle[1][1],
                                            angle=rectangle[2]))

        tissue_objects = TissueObjects(mask, rectangles)

    image = F.pack_atlas(image, tissue_objects)
    return image, tissue_objects
