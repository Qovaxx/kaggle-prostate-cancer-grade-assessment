from typing import (
    Tuple,
    List,
    Optional,
    NamedTuple
)

import numpy as np

IMAGE_TYPE = np.ndarray
MASK_TYPE = Optional[np.ndarray]

CV2_RECT_TYPE = Tuple[Tuple[float, float], Tuple[float, float], float]
RECTPACK_RECT_TYPE = Tuple[int, int, int, int, int, int]

SLICE_TYPE = List[bool]

class Contour(NamedTuple):
    mask: np.ndarray
    rectangle: CV2_RECT_TYPE
