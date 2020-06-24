from typing import (
    Dict,
    Tuple
)

import cv2
import numpy as np


def draw_overlay_mask(image: np.ndarray, mask: np.ndarray, color_map: Dict[int, Tuple[int, int, int]],
                      alpha: float = 0.6) -> np.ndarray:
    assert image.shape[2] == 3, "Image must be in RGB format"
    assert len(mask.shape) == 2, "Mask must be in grayscale format"
    assert len(next(iter(color_map.items()))[1]) == 3, "Colors must be in RGB format"
    masked = image.copy()
    contrast_mask = np.zeros(shape=(*mask.shape[:2], 3), dtype=masked.dtype)
    unique_classes = np.unique(mask)
    for segmentation_class in unique_classes:
        contrast_mask[mask == segmentation_class] = color_map[segmentation_class]

    return cv2.addWeighted(src1=masked, alpha=alpha, src2=contrast_mask, beta=(1-alpha), dst=masked, gamma=0)
