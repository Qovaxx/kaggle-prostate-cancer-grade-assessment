from typing import (
    Dict,
    Optional,
    NoReturn,
    Union
)

import albumentations as A
from torch.utils.data import Dataset

from . import functional as F
from ...read import TIFFReader
from ...base import BasePhaseSplitter
from ...split import CleanedPhaseSplitter
from ....spacer import SpaceConverter
from ....phase import Phase
from ....settings import MAX_CM_RESOLUTION


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


class PSGAPatchSequenceClassificationDataset(Dataset):

    def __init__(self, path: str, micron_tile_size: int = 231,
                 fold: int = 0, phase: str = "train",
                 crop_filter_threshold: float = 0.1,
                 transforms: Optional[A.Compose] = None,
                 phase_splitter: Optional[BasePhaseSplitter] = CleanedPhaseSplitter()) -> NoReturn:

        self._reader = TIFFReader(path)
        self._micron_tile_size = micron_tile_size
        self._pixel_tile_size = SpaceConverter(cm_resolution=MAX_CM_RESOLUTION).microns_to_pixels(micron_tile_size)
        self._crop_filter_threshold = crop_filter_threshold
        self._transforms = transforms

        if phase_splitter:
            phase_splitter.split(self._reader)

        meta = self._reader.meta
        rng = range(len(meta))
        phase = Phase[phase.upper()]
        if phase == Phase.TRAIN:
            phase_indices = [i for i in rng if meta[i]["fold"] != fold and meta[i]["phase"] != Phase.TEST]
        elif phase == Phase.VAL:
            phase_indices = [i for i in rng if meta[i]["fold"] == fold and meta[i]["phase"] != Phase.TEST]
        elif phase == Phase.TEST:
            phase_indices = [i for i in rng if meta[i]["phase"] == phase]
        else:
            raise ValueError("Available phases are: train, val, test")

        self._index_map = dict(enumerate(phase_indices))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, item: int): # -> Dict[str, Union[Tensor, int]]:
        record = self._reader.get(self._index_map[item], read_mask=False, read_visualization=False)
        spacer = SpaceConverter(cm_resolution=record.additional["x_resolution"])
        pixel_tile_size = spacer.microns_to_pixels(self._micron_tile_size)

        tiles, coordinates = F.cut_tiles(record.image, tile_size=pixel_tile_size, remove_empty_tiles=True,
                                         calculate_coordinates=True, filter_empty_threshold=self._crop_filter_threshold)


        return {"target": record.label}
