from typing import (
    Optional,
    Tuple
)

import numpy as np

from . import functional as F


def crop_external_background(image: np.ndarray, bbox: Optional[np.ndarray] = None,
                             background_value: int = 255) -> Tuple[np.ndarray, np.ndarray]:
    if bbox is None:
        height_indices = (image.min(axis=(1, 2)) < background_value).nonzero()[0]
        width_indices = (image.min(axis=(0, 2)) < background_value).nonzero()[0]
        bbox = np.array([[width_indices[0], height_indices[0]], [width_indices[-1], height_indices[-1]]])
        bbox = np.int0(bbox)

    image = F.crop_bbox(image, bbox)

    return image, bbox
