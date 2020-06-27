from typing import (
    Optional,
    Tuple
)

import numpy as np

from .entity import (
    BBox,
    Slice2D
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
