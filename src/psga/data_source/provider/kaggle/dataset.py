from typing import (
    Dict,
    Optional,
    ClassVar,
    List,
    NoReturn,
    Union
)

import torch
import albumentations as A
import numpy as np
from torch.utils.data import Dataset

from .utils import (
    zoom_tiles,
    bin_label,
    balanced_subsample
)
from ...read import TIFFReader
from ...base import BasePhaseSplitter
from ...split import CleanedPhaseSplitter
from ....spacer import SpaceConverter
from ....phase import Phase
from ....transforms.slicer import TilesSlicer
from ....grade import (
    CancerGradeSystem,
    mask_to_gleason_score
)
from ....utils.inout import (
    load_pickle,
    save_pickle,
    load_file
)
from ....settings import (
    DATA_DIRPATH,
    EMPTY_MASKS_PATH,
    MAX_CM_RESOLUTION
)


class _BasePSGATileDataset(Dataset):

    available_phases: ClassVar[List[str]] = ["train", "val", "test"]

    def __init__(self, path: str,
                 phase: str = "train", fold: int = 0,
                 micron_tile_size: int = 231,
                 crop_emptiness_degree: float = 0.9,
                 label_binning: bool = True,
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

        self._crop_emptiness_degree = crop_emptiness_degree
        self._label_binning = label_binning

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

    def __init__(self, tiles_intersection: float = 0.5, subsample_tiles_count: Optional[int] = None,
                 balance_subsample: bool = False, *args, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)

        self._tiles_intersection = tiles_intersection
        self._subsample_tiles_count = subsample_tiles_count
        self._balance_subsample = balance_subsample

        empty_masks = load_file(str(EMPTY_MASKS_PATH))
        phase_indices = [i for i in self._phase_indices
                         if self._reader.meta[i]["additional"]["data_provider"] == "radboud"
                         and self._reader.meta[i]["name"] not in empty_masks
                         and self._reader.meta[i]["mask"] is not None]
        if self._phase == Phase.VAL:
            phase_indices = phase_indices[:20]
        self._index_map = dict(enumerate(phase_indices))
        self._grader = CancerGradeSystem()

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, item: int) -> Dict[str, Union[torch.Tensor, np.ndarray, int]]:
        record = self._reader.get(self._index_map[item], read_mask=True, read_visualization=False)
        image = record.image
        mask = record.mask
        spacer = SpaceConverter(cm_resolution=record.additional["x_resolution"])
        actual_tile_size = spacer.microns_to_pixels(self._micron_tile_size)

        if self._image_transforms:
            transformed = self._image_transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        slicer = TilesSlicer(actual_tile_size, intersection=self._tiles_intersection,
                             remove_empty_tiles=True, emptiness_degree=self._crop_emptiness_degree)
        image_tiles, non_empty_tiles_indices = slicer(image)
        slicer._fill_value = 0
        mask_tiles, _ = slicer(mask, non_empty_tiles_indices)

        labels = [mask_to_gleason_score(x) for x in mask_tiles]
        labels = [self._grader.gleason_to_isup(x.major_value, x.minor_value) for x in labels]

        selected_indices = np.asarray(list(range(image_tiles.shape[0])))
        if self._subsample_tiles_count:
            if self._balance_subsample:
                selected_indices = np.asarray(balanced_subsample(labels, count=self._subsample_tiles_count))
            else:
                selected_indices = np.random.choice(list(range(image_tiles.shape[0])),
                                                    size=self._subsample_tiles_count, replace=False)

        image_tiles = image_tiles[selected_indices]
        labels = [labels[i] for i in selected_indices]

        if actual_tile_size != self._pixel_tile_size:
            image_tiles = zoom_tiles(image_tiles, shape=(self._pixel_tile_size, self._pixel_tile_size))

        if self._crop_transforms:
            image_tiles = [self._crop_transforms(image=tile)["image"] for tile in image_tiles]
            lib = np if isinstance(image_tiles[0], np.ndarray) else torch
            image_tiles = lib.stack(image_tiles)

        items = item * 10**4 + selected_indices # 10**4 it's dirty hack to generate unique crop index
        if self._label_binning:
            labels = [bin_label(x, classes=len(self._grader.isup_grades)) for x in labels]

        return {"item": items, "image": image_tiles, "target": labels}

    @staticmethod
    def fast_collate_fn(batch: List[Dict[str, Union[torch.Tensor, np.ndarray, int]]]) -> Dict[str, torch.Tensor]:
        collated_images = [x["image"] for x in batch]
        if isinstance(batch[0]["image"][0], np.ndarray):
            image = torch.from_numpy(np.concatenate(collated_images))
        else:
            image = torch.cat(collated_images)

        target = torch.from_numpy(np.concatenate([x["target"] for x in batch]))
        item = torch.from_numpy(np.concatenate([x["item"] for x in batch]))

        return {"item": item, "image": image, "target": target}



class PSGATileSequenceClassificationDataset(_BasePSGATileDataset):

    def __init__(self, *args, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)
        self._index_map = dict(enumerate(self._phase_indices))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, item: int) -> Dict[str, Union[torch.Tensor, np.ndarray, int]]:
        record = self._reader.get(self._index_map[item], read_mask=False, read_visualization=False)
        image = record.image
        data_provider = 1 if record.additional["data_provider"] == "radboud" else 0
        spacer = SpaceConverter(cm_resolution=record.additional["x_resolution"])
        actual_tile_size = spacer.microns_to_pixels(self._micron_tile_size)

        if self._image_transforms:
            image = self._image_transforms(image=image)["image"]

        slicer = TilesSlicer(actual_tile_size, remove_empty_tiles=True, emptiness_degree=self._crop_emptiness_degree)
        tiles, _ = slicer(image)

        if actual_tile_size != self._pixel_tile_size:
            tiles = zoom_tiles(tiles, shape=(self._pixel_tile_size, self._pixel_tile_size))

        if self._crop_transforms:
            tiles = [self._crop_transforms(image=tile)["image"] for tile in tiles]
            lib = np if isinstance(tiles[0], np.ndarray) else torch
            tiles = lib.stack(tiles)

        label = record.label
        if self._label_binning:
            label = bin_label(label, classes=len(CancerGradeSystem().isup_grades))

        return {"item": item, "image": tiles, "target": label, "data_provider": data_provider}

    @staticmethod
    def fast_collate_fn(batch: List[Dict[str, Union[torch.Tensor, np.ndarray, int]]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
