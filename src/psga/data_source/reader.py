from tifffile import imread
from typing_extensions import final

import cv2

from .base import BaseReader
from .record import Record



@final
class TIFFReader(BaseReader):

    def get(self, index: int, read_mask: bool = True,
            read_visualization: bool = False) -> Record:
        record = self.meta[index]
        record = Record(**record)
        record.image = imread(self.images_path.parent / record.image)

        if read_mask:
            record.mask = imread(self.masks_path.parent / record.mask)

        if read_visualization:
            record.visualization = cv2.imread(str(self.visualizations_path.parent / record.visualization),
                                              flags=cv2.COLOR_BGR2RGB)

        return record


@final
class JPEGReader(BaseReader):

    def get(self, index: int) -> Record:
        raise NotImplementedError
