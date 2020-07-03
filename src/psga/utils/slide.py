from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.io import MultiImage


def get_layer_safely(slide: MultiImage, layer: int, is_mask: bool = False,
                     across_layer_scale: int = 4) -> Optional[np.ndarray]:
    filename = Path(slide.filename).stem
    if is_mask:
        for current_layer in range(layer, 3):
            try:
                mask = slide[current_layer][..., 0]
                if current_layer != layer:
                    scale = across_layer_scale ** (current_layer - layer)
                    mask = cv2.resize(mask, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                return mask
            except Exception as e:
                print(f"{filename}[{current_layer}] - {e}")
        return None

    else:
        try:
            return slide[layer]
        except Exception as e:
            print(f"{filename}[{layer}] - {e}")
            return None
