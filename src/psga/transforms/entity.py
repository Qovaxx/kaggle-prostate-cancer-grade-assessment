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
class Intermediates(BaseEntity):
    external_bbox: Optional[BBox] = field(default=None)
    inner_slice: Optional[Slice2D] = field(default=None)

    def rescale(self, scale: int) -> NoReturn:
        self.external_bbox.rescale(scale)
        self.inner_slice.rescale(scale)
