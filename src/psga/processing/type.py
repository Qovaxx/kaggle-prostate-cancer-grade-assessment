from typing import (
    Tuple,
    NamedTuple
)

import numpy as np

CV2_RECT_TYPE = Tuple[Tuple[float, float], Tuple[float, float], float]
RECTPACK_RECT_TYPE = Tuple[int, int, int, int, int, int]


class Contour(NamedTuple):
    mask: np.ndarray
    rectangle: CV2_RECT_TYPE
