from typing import (
    Dict,
    Optional,
    ClassVar,
    List,
    NoReturn,
    Union
)

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset

from ...read import TIFFReader
from ...base import BasePhaseSplitter
from ...split import CleanedPhaseSplitter
from ....spacer import SpaceConverter
from ....phase import Phase
from ....transforms.slicer import TilesSlicer
from ....utils.inout import (
    load_pickle,
    save_pickle
)
from ....settings import (
    DATA_DIRPATH,
    MAX_CM_RESOLUTION
)


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()



class _BasePSGATileDataset(Dataset):

    available_phases: ClassVar[List[str]] = ["train", "val", "test"]

    def __init__(self, path: str,
                 phase: str = "train", fold: int = 0,
                 micron_tile_size: int = 231,
                 image_transforms: Optional[A.Compose] = None,
                 crop_transforms: Optional[A.Compose] = None,
                 phase_splitter: Optional[BasePhaseSplitter] = CleanedPhaseSplitter(),
                 split_filename: str = "phase_splitted_meta",
                 *args, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)

        self._reader = TIFFReader(path)

        assert phase in self.available_phases, f"Available phases are: {self.available_phases}"
        self._phase = Phase[phase.upper()]
        self._fold = fold

        self._micron_tile_size = micron_tile_size
        self._pixel_tile_size = SpaceConverter(cm_resolution=MAX_CM_RESOLUTION).microns_to_pixels(micron_tile_size)

        self._image_transforms = image_transforms
        self._crop_transforms = crop_transforms

        if phase_splitter:
            file_path = (DATA_DIRPATH / split_filename).with_suffix(".pkl")
            if file_path.exists():
                meta = load_pickle(str(file_path))
                self._reader.meta = meta
            else:
                meta = phase_splitter.split(self._reader.meta)
                save_pickle(meta, path=str(file_path))

        meta = self._reader.meta
        rng = range(len(meta))
        if self._phase == Phase.TRAIN:
            self._phase_indices = [i for i in rng if meta[i]["fold"] != self._fold and meta[i]["phase"] != Phase.TEST]
        elif self._phase == Phase.VAL:
            self._phase_indices = [i for i in rng if meta[i]["fold"] == self._fold and meta[i]["phase"] != Phase.TEST]
        else:
            self._phase_indices = [i for i in rng if meta[i]["phase"] == self._phase]


class PSGATileMaskedClassificationDataset(_BasePSGATileDataset):

    def __init__(self, *args, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)

        phase_indices = [i for i in self._phase_indices
                         if self._reader.meta[i]["additional"]["data_provider"] == "radboud"]
        # Remove empty masks



        a = 4





class PSGATileSequenceClassificationDataset(_BasePSGATileDataset):

    def __init__(self, _crop_emptiness_degree: float = 0.9, *args, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)
        self._crop_emptiness_degree = _crop_emptiness_degree
        self._index_map = dict(enumerate(self._phase_indices))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, item: int) -> Dict[str, Union[torch.Tensor, np.ndarray, int]]:
        record = self._reader.get(self._index_map[item], read_mask=False, read_visualization=False)
        image = record.image
        data_provider = 1 if record.additional["data_provider"] == "radboud" else 0
        spacer = SpaceConverter(cm_resolution=record.additional["x_resolution"])
        pixel_tile_size = spacer.microns_to_pixels(self._micron_tile_size)

        if self._image_transforms:
            image = self._image_transforms(image=image)["image"]

        slicer = TilesSlicer(pixel_tile_size, remove_empty_tiles=True, emptiness_degree=self._crop_emptiness_degree)
        tiles, _ = slicer(image)

        if pixel_tile_size != self._pixel_tile_size:
            tiles = np.asarray([
                cv2.resize(tile, dsize=(self._pixel_tile_size, self._pixel_tile_size), interpolation=cv2.INTER_LANCZOS4)
                for tile in tiles
            ])

        if self._crop_transforms:
            tiles = [self._crop_transforms(image=tile)["image"] for tile in tiles]
            lib = np if isinstance(tiles[0], np.ndarray) else torch
            tiles = lib.stack(tiles)

        return {"tiles": tiles, "isup_grade": record.label, "data_provider": data_provider}
