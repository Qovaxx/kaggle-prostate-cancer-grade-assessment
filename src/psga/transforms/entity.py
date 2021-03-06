from abc import (
    ABC,
    abstractmethod
)
from dataclasses import (
    dataclass,
    field
)
from typing import (
    NoReturn,
    Tuple,
    Optional,
    List
)

import cv2
import numpy as np


class BaseEntity(ABC):

    @abstractmethod
    def rescale(self, scale: int) -> NoReturn:
        ...


@dataclass
class BBox(BaseEntity):
    x: int
    y: int
    width: int
    height: int

    def rescale(self, scale: int) -> NoReturn:
        self.x *= scale
        self.y *= scale
        self.width *= scale
        self.height *= scale


@dataclass
class Slice2D(BaseEntity):
    rows: List[bool]
    columns: List[bool]

    def rescale(self, scale: int) -> NoReturn:
        self.rows = self._rescale(self.rows, new_size=len(self.rows) * scale)
        self.columns = self._rescale(self.columns, new_size=len(self.columns) * scale)

    @staticmethod
    def _rescale(slice: List[bool], new_size: int) -> List[bool]:
        slice_vector = np.asarray(slice).astype(np.uint8)
        slice_vector = cv2.resize(slice_vector, dsize=(1, new_size), interpolation=cv2.INTER_NEAREST)
        return slice_vector.astype(np.bool).ravel().tolist()


@dataclass
class CV2Rectangle(BaseEntity):
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float

    def to_d4_bbox(self) -> Optional[BBox]:
        if self.angle % 90 == 0:
            return BBox(x=int(self.center_x - (self.width / 2)),
                        y=int(self.center_y - (self.height / 2)),
                        width=int(self.width),
                        height=int(self.height))
        return None

    def to_cv2_rect(self) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        return ((self.center_x, self.center_y), (self.width, self.height), self.angle)

    def rescale(self, scale: int) -> NoReturn:
        self.center_x *= scale
        self.center_y *= scale
        self.width *= scale
        self.height *= scale


@dataclass
class RectPackRectangle(BaseEntity):
    bin: int
    x: int
    y: int
    width: int
    height: int
    index: int

    def rescale(self, scale: int) -> NoReturn:
        self.x *= scale
        self.y *= scale
        self.width *= scale
        self.height *= scale


@dataclass
class TissueObjects(BaseEntity):
    mask: np.ndarray
    cv2_rectangles: List[CV2Rectangle]
    bin_shape: Tuple[int, int]
    rectpack_rectangles: List[RectPackRectangle]

    def rescale(self, scale: int, **kwargs) -> NoReturn:
        mask_shape = tuple(np.asarray(self.mask.shape) * scale)
        self.mask = cv2.resize(self.mask, dsize=mask_shape[::-1], interpolation=cv2.INTER_NEAREST)
        bin_shape = [0, 0]
        for index, cv2_rect in enumerate(self.cv2_rectangles):
            cv2_rect.rescale(scale)
            rectpack_rect = list(filter(lambda x: x.index == index, self.rectpack_rectangles))[0]
            rectpack_rect.rescale(scale)
            # Rectpack rectangles can be rotated
            if abs(cv2_rect.width - rectpack_rect.width) < abs(cv2_rect.width - rectpack_rect.height):
                rectpack_rect.width = int(cv2_rect.width)
                rectpack_rect.height = int(cv2_rect.height)
            else:
                rectpack_rect.width = int(cv2_rect.height)
                rectpack_rect.height = int(cv2_rect.width)

            max_width = rectpack_rect.x + rectpack_rect.width
            max_height = rectpack_rect.y + rectpack_rect.height
            if max_height >= bin_shape[0]:
                bin_shape[0] = max_height
            if max_width >= bin_shape[1]:
                bin_shape[1] = max_width
        exprected_bin_shape = tuple(np.asarray(self.bin_shape) * scale)
        self.bin_shape = (max(bin_shape[0], exprected_bin_shape[0]), max(bin_shape[1], exprected_bin_shape[1]))


@dataclass
class Intermediates(BaseEntity):
    external_bbox: Optional[BBox] = field(default=None)
    inner_slice: Optional[Slice2D] = field(default=None)
    rough_tissue_objects: Optional[TissueObjects] = field(default=None)
    clear_mask: Optional[np.ndarray] = field(default=None)
    precise_tissue_objects: Optional[TissueObjects] = field(default=None)

    def rescale(self, scale: int) -> NoReturn:
        self.external_bbox.rescale(scale)
        self.inner_slice.rescale(scale)
        self.rough_tissue_objects.rescale(scale)
        self.precise_tissue_objects.rescale(scale)

        shape = tuple(np.asarray(self.clear_mask.shape) * scale)
        self.clear_mask = cv2.resize(self.clear_mask, dsize=shape[::-1], interpolation=cv2.INTER_NEAREST)
