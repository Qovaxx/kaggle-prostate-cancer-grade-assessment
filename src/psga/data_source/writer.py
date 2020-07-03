from dataclasses import asdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    NoReturn
)

from PIL import Image
from typing_extensions import final

from .base import BaseWriter
from .record import Record
from ..utils import tiff


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
        visualization_path = (self.visualizations_path / relative_path).with_suffix(".jpg")

        image_path.parent.mkdir(parents=True, exist_ok=True)
        tiff.save_tiff_image(record.image, path=str(image_path), quality=self._quality, tile_size=self._tile_size)

        if record.mask is not None:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            tiff.save_tiff_mask(record.mask, path=str(mask_path), tile_size=self._tile_size)

            visualization_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(record.visualization).save(str(visualization_path), quality=self._quality, subsampling=0)

        meta = asdict(record)
        meta["image"] = self._to_relative(image_path)
        meta["mask"] = self._to_relative(mask_path) if record.mask is not None else None
        meta["visualization"] = self._to_relative(visualization_path) if record.visualization is not None else None
        return meta


@final
class JPEGWriter(BaseWriter):

    def _put(self, record: Record) -> Dict[str, Any]:
        raise NotImplementedError
