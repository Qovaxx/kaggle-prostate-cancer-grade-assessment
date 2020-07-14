from typing import (
    Dict,
    Optional,
    NoReturn,
    Union
)

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset

from . import functional as F
from ...read import TIFFReader
from ...base import BasePhaseSplitter
from ...split import CleanedPhaseSplitter
from ....spacer import SpaceConverter
from ....phase import Phase
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


class PSGAPatchSequenceClassificationDataset(Dataset):

    def __init__(self, path: str, micron_tile_size: int = 231,
                 fold: int = 0, phase: str = "train",
                 crop_filter_threshold: float = 0.1,
                 image_transforms: Optional[A.Compose] = None,
                 crop_transforms: Optional[A.Compose] = None,
                 phase_splitter: Optional[BasePhaseSplitter] = CleanedPhaseSplitter()):

        self._reader = TIFFReader(path)
        self._micron_tile_size = micron_tile_size
        self._pixel_tile_size = SpaceConverter(cm_resolution=MAX_CM_RESOLUTION).microns_to_pixels(micron_tile_size)
        self._crop_filter_threshold = crop_filter_threshold
        self._image_transforms = image_transforms
        self._crop_transforms = crop_transforms

        if phase_splitter:
            file_path = DATA_DIRPATH / "phase_splitted_meta.pkl"
            if file_path.exists():
                meta = load_pickle(str(file_path))
                self._reader.meta = meta
            else:
                meta = phase_splitter.split(self._reader.meta)
                save_pickle(meta, path=str(file_path))

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

    def __getitem__(self, item: int) -> Dict[str, Union[torch.Tensor, np.ndarray, int]]:
        record = self._reader.get(self._index_map[item], read_mask=False, read_visualization=False)
        image = record.image
        data_provider = 1 if record.additional["data_provider"] == "radboud" else 0
        spacer = SpaceConverter(cm_resolution=record.additional["x_resolution"])
        pixel_tile_size = spacer.microns_to_pixels(self._micron_tile_size)

        if self._image_transforms:
            image = self._image_transforms(image=image)["image"]

        tiles, coords = F.cut_tiles(image, tile_size=pixel_tile_size,
                                    remove_empty_tiles=True, return_normed_coords=True,
                                    filter_empty_threshold=self._crop_filter_threshold)

        if pixel_tile_size != self._pixel_tile_size:
            resize_it = lambda iterable, inter: np.asarray(
                [cv2.resize(x, dsize=(self._pixel_tile_size, self._pixel_tile_size), interpolation=inter) for x in iterable]
            )
            tiles = resize_it(tiles, inter=cv2.INTER_LANCZOS4)
            coords = resize_it(coords, inter=cv2.INTER_NEAREST)

        if self._crop_transforms:
            transformed_crops = list()
            transformed_coords = list()

            for tile, coord in zip(tiles, coords):
                transformed = self._crop_transforms(image=tile, mask=coord)
                transformed_crops.append(transformed["image"])

                coord = transformed["mask"]
                if ToTensorV2 in map(type, self._crop_transforms.transforms.transforms):
                    coord = coord.permute(2, 0, 1).float()
                transformed_coords.append(coord)

            lib = np if isinstance(transformed_crops[0], np.ndarray) else torch
            tiles = lib.stack(transformed_crops)
            coords = lib.stack(transformed_coords)

        return {"tiles": tiles, "coords": coords, "target": record.label, "data_provider": data_provider}
