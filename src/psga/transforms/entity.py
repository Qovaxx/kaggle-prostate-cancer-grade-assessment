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
class Rectangle(BaseEntity):
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
class TissueObject(BaseEntity):
    mask: np.ndarray
    rectangle: Rectangle

    def rescale(self, scale: int, **kwargs) -> NoReturn:
        shape = tuple(np.asarray(self.mask.shape) * scale)
        mask = self.mask.astype(np.uint8)
        self.mask = cv2.resize(mask, dsize=shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.bool)
        self.rectangle.rescale(scale)

@dataclass
class Intermediates(BaseEntity):
    external_bbox: Optional[BBox] = field(default=None)
    inner_slice: Optional[Slice2D] = field(default=None)
    tissue_objects: Optional[List[TissueObject]] = field(default=None)

    def rescale(self, scale: int) -> NoReturn:
        self.external_bbox.rescale(scale)
        self.inner_slice.rescale(scale)
        for obj in self.tissue_objects:
            obj.rescale(scale)
