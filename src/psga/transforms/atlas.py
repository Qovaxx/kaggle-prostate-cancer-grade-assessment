from typing import (
    List,
    Tuple,
    Optional
)

import cv2
import numpy as np

from . import functional as F
from .entity import (
    TissueObject,
    Rectangle
)


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def convert_to_atlas(image: np.ndarray, tissue_objects: Optional[List[TissueObject]] = None,
                     min_contour_area: int = 200,
                     background_value: int = 255) -> Tuple[np.ndarray, List[TissueObject]]:
    if tissue_objects is None:
        gray = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
        _, not_background_mask = cv2.threshold(gray, thresh=background_value - 1, maxval=image.max(),
                                               type=cv2.THRESH_BINARY_INV)
        found_contours, _ = cv2.findContours(not_background_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        tissue_objects = list()

        for contour in found_contours:
            if cv2.contourArea(contour) >= min_contour_area:
                contour_mask = np.full(image.shape[:2], fill_value=0, dtype=np.uint8)
                cv2.drawContours(contour_mask, contours=[contour], contourIdx=0, color=1, thickness=-1)
                rectangle = cv2.minAreaRect(points=contour)
                rectangle = Rectangle(center_x=rectangle[0][0],
                                      center_y=rectangle[0][1],
                                      width=rectangle[1][0],
                                      height=rectangle[1][1],
                                      angle=rectangle[2])
                tissue_objects.append(TissueObject(mask=contour_mask.astype(np.bool), rectangle=rectangle))

    image = F.pack_atlas(image, tissue_objects)
    return image, tissue_objects
