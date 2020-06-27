from typing import (
    Optional,
    Tuple,
    List
)

import cv2
import numpy as np
from skimage import morphology

from . import functional as F
from .preprocessing import reduce_external_background, reduce_inner_background, crop_roi
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



def compose_pre_processing(image: IMAGE_TYPE, mask: MASK_TYPE,
                           external_bbox: Optional[np.ndarray] = None,
                           inner_row_slice: Optional[SLICE_TYPE] = None,
                           inner_column_slice: Optional[SLICE_TYPE] = None,
                           roi_rectangle: Optional[CV2_RECT_TYPE] = None,
                           ) -> Tuple[IMAGE_TYPE, MASK_TYPE]:

    image, mask, external_bbox = reduce_external_background(image, mask, bbox=external_bbox)
    image, mask, inner_row_slice, inner_column_slice = \
        reduce_inner_background(image, mask, row_slice=inner_row_slice, column_slice=inner_column_slice)
    image, mask, roi_rectangle = crop_roi(image, mask, rectangle=roi_rectangle)









    a = 4




