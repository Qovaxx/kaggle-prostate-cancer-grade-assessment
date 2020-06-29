import io
import matplotlib.pyplot as plt
from typing import (
    Dict,
    Optional,
    List,
    Tuple,
    Union
)
from matplotlib.patches import Patch

import cv2
import numpy as np

RGB_TYPE = Tuple[Union[int, float], Union[int, float], Union[int, float]]


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


def plot_meta(image: np.ndarray, title_text: str,
              color_map: Dict[int, RGB_TYPE], classname_map: Dict[int, str],
              dpi: int = 300, show_keys: Optional[List[int]] = None) -> np.ndarray:

    assert len(set(color_map.keys()).difference(set(classname_map.keys()))) == 0,\
        "Color map and class name map must contain the same keys"
    keys = color_map.keys() if show_keys is None else show_keys
    patches = [Patch(color=color_map[x], label=classname_map[x]) for x in keys]

    plt.figure()
    plt.title(title_text)
    plt.legend(handles=patches, bbox_to_anchor=(1.04, 0.5), loc="center left", prop={"size": 6})
    plt.imshow(image)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi)
    plt.close()
    buffer.seek(0)
    eda = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    buffer.close()

    return cv2.cvtColor(cv2.imdecode(eda, 1), cv2.COLOR_BGR2RGB)
