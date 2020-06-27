import numpy as np


def crop_bbox(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    return image[bbox[0][1]: bbox[1][1], bbox[0][0]: bbox[1][0]]
