from typing import (
    Dict,
    Optional,
    List,
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


def crop_tissue_roi(image: np.ndarray, additional_images: Optional[List[np.ndarray]] = None,
                    min_background_value: int = 255) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    if additional_images is not None:
        for additional_image in additional_images:
            assert image.shape[:2] == additional_image.shape[:2]

    height_indices = (image.min(axis=(1, 2)) < min_background_value).nonzero()[0]
    width_indices = (image.min(axis=(0, 2)) < min_background_value).nonzero()[0]
    crop_it = lambda x: np.take(np.take(x, indices=height_indices, axis=0), indices=width_indices, axis=1)
    cropped_image = crop_it(image)

    if additional_images is None:
        return cropped_image, None
    return cropped_image, list(map(crop_it, additional_images))
