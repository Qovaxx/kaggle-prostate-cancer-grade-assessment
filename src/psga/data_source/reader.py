from typing import NoReturn

import tifffile
from typing_extensions import final

from .base import BaseReader
from .record import Record


@final
class TIFFReader(BaseReader):

    def get(self, index: int) -> Record:
        record = self.meta[index]


        a = 4


    def num_images(self) -> int:
        return 31


@final
class JPEGReader(BaseReader):

    def get(self, index: int) -> Record:
        raise NotImplementedError
