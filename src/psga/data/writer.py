from pathlib import Path
from typing import NoReturn

from PIL import Image

from .base import BaseWriter
from .record import Record

from time import time
import pyvips
from .pyvips_ import to_numpy, from_numpy
from tifffile import imread
from .pyvips_ import imread as vimread
from collections import Counter
from pyvips.enums import ForeignTiffCompression
from skimage.io import MultiImage
import numpy as np
import tifffile


class TIFFWriter(BaseWriter):

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


        # Counter({True: 2353793031, False: 127513593}) simple
        # Counter({True: 2471862811, False: 9443813}) Q=100
        # Counter({True: 2413767353, False: 67539271}) Q=90
        vi = from_numpy(record.image)
        vi.write_to_file(str(image_path.with_suffix(".tiff")),
                         bigtiff=True, compression=ForeignTiffCompression.JPEG, Q=90,
                         tile=True, tile_width=512, tile_height=512,
                         rgbjpeg=True)

        # ForeignTiffCompression.WEBP
        vi_mask = from_numpy(np.expand_dims(record.mask, axis=2))
        vi_mask.write_to_file(str(mask_path.with_suffix(".tiff")),
                         bigtiff=True, compression=ForeignTiffCompression.LZW,
                         tile=True, tile_width=512, tile_height=512, rgbjpeg=False)

        # recm = MultiImage("/data/processed/prostate-cancer-grade-assessment/data/masks/031f5ef5b254fbacd6fbd279ebfe5cc0_mask.tiff")[0]
        start = time()
        recm = tifffile.imread(str(mask_path.with_suffix(".tiff")))
        print(time() - start)


        start = time()
        recovered1 = tifffile.imread(image_path.with_suffix(".tiff"))
        print(time() - start)

        image_path.parent.mkdir(parents=True, exist_ok=True)
        # Image.fromarray(record.image).save(str(image_path), quality=self._quality, subsampling=self._subsampling)

        if record.mask is not None:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            # Image.fromarray(record.mask).save(str(mask_path))

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
