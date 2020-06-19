from pathlib import Path
from typing import NoReturn

from PIL import Image

from .base import BaseWriter
from .record import Record


class JPEGWriter(BaseWriter):

    def __init__(self, path: str, quality: int = 100, subsampling: int = 0) -> NoReturn:
        super().__init__(path)
        self._attributes = list()
        self._quality = quality
        self._subsampling = subsampling

    def _put(self, record: Record) -> NoReturn:
        relative_path = Path(str(record.label)) / record.name
        image_path = (self.images_path / relative_path).with_suffix(".jpg")
        mask_path = (self.masks_path / relative_path).with_suffix(".png")
        eda_path = (self.eda_path / relative_path).with_suffix(".jpg")

        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(record.image).save(str(image_path), quality=self._quality, subsampling=self._subsampling)

        if record.mask is not None:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(record.mask).save(str(mask_path))

        eda_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(record.eda).save(str(eda_path), quality=self._quality, subsampling=self._subsampling)

        self._attributes.append({
            "image_path": self._to_relative(image_path),
            "mask_path": self._to_relative(mask_path) if record.mask is not None else None,
            "eda_path": self._to_relative(eda_path),
            "name": record.name,
            "label": record.label,
            "fold": record.fold,
            "phase": record.phase.value if record.phase is not None else None,
            "additional": record.additional
        })

    @staticmethod
    def _to_relative(path: Path) -> str:
        return str(path.relative_to(path.parent.parent.parent))
