from dataclasses import asdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    NoReturn
)

from PIL import Image
from pyvips.enums import ForeignTiffCompression
from typing_extensions import final

from .base import BaseWriter
from .record import Record
from ..utils import vips

import os
@final
class TIFFWriter(BaseWriter):

    def __init__(self, path: str, quality: int = 90, tile_size: int = 512) -> NoReturn:
        super().__init__(path)
        self._quality = quality
        self._tile_size = tile_size

    def _put(self, record: Record) -> Dict[str, Any]:
        relative_path = Path(str(record.label)) / record.name
        image_path = (self.images_path / relative_path).with_suffix(".tiff")
        mask_path = (self.masks_path / relative_path).with_suffix(".tiff")
        eda_path = (self.eda_path / relative_path).with_suffix(".jpg")

        image_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists():
            os.remove(str(image_path))
        vips.from_numpy(record.image).write_to_file(str(image_path),
                                                    bigtiff=True, compression=ForeignTiffCompression.JPEG, Q=self._quality,
                                                    tile=True, tile_width=self._tile_size, tile_height=self._tile_size,
                                                    rgbjpeg=True)
        if record.mask is not None:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if mask_path.exists():
                os.remove(str(mask_path))
            vips.from_numpy(record.mask).write_to_file(str(mask_path),
                                                       bigtiff=True, compression=ForeignTiffCompression.LZW,
                                                       tile=True, tile_width=self._tile_size, tile_height=self._tile_size,
                                                       rgbjpeg=False)

            eda_path.parent.mkdir(parents=True, exist_ok=True)
            if eda_path.exists():
                os.remove(str(eda_path))
            Image.fromarray(record.visualization).save(str(eda_path), quality=self._quality, subsampling=0)

        meta = asdict(record)
        meta["utils"] = self._to_relative(image_path)
        meta["mask"] = self._to_relative(mask_path) if record.mask is not None else None,
        meta["eda"] = self._to_relative(eda_path) if record.visualization is not None else None,
        return meta


@final
class JPEGWriter(BaseWriter):

    def _put(self, record: Record) -> Dict[str, Any]:
        raise NotImplementedError
