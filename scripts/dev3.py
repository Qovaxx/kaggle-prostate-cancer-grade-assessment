import os
import warnings
from typing import Dict, Tuple, List, Optional, Callable, NoReturn, Union

import cv2
from typing_extensions import Final

import numpy as np
import pandas as pd
import torch
from albumentations import (
    Compose,
    Transpose,
    VerticalFlip,
    HorizontalFlip,
    Resize,
)

# try:
#     from openslide import OpenSlide
# except ImportError as e:
#     warnings.warn(f"Error {e} during importing openslide library")

from ok_constants.constants import LABEL_MAPPING_NAME
from ok_constants.custom_types import NUMPY_TYPE
from ok_tasks.kaggle.prostate.get_tiles import Tile
from torch.utils.data import Dataset

from ...registry import DATASETS
from ...utils import (
    LOADER_MAPPING,
    load_label_mapping,
    float_target_transform,
)


MICRONS_PER_CM: Final = 10000
SIZE_TYPE = Union[int, Tuple[int, int]]


class SpaceConverter(object):
    def __init__(self, cm_resolution: float) -> NoReturn:
        self._pixel_spacing = 1 / (float(cm_resolution) / MICRONS_PER_CM)

    def _apply(
        self, input: SIZE_TYPE, func: Callable[[SIZE_TYPE], SIZE_TYPE]
    ) -> SIZE_TYPE:
        if isinstance(input, int):
            return func(input)
        else:
            return func(input[0]), func(input[1])

    def pixels_to_microns(self, pixels_size: SIZE_TYPE) -> SIZE_TYPE:
        to_microns = lambda x: round(self._pixel_spacing * (x - 1))
        return self._apply(pixels_size, to_microns)

    def microns_to_pixels(self, microns_size: SIZE_TYPE) -> SIZE_TYPE:
        to_pixels = lambda x: round(x / self._pixel_spacing + 1)
        return self._apply(microns_size, to_pixels)


@DATASETS.register_module
class ProstateCancerTileDataset(Dataset):
    def __init__(
        self,
        file_name: str,
        label_column: str,
        path_column: str = "image_id",
        img_root: str = "./",
        mask_root: str = "./",
        use_crop_transform: bool = False,
        transform=None,
        label_mapping_path: str = None,
        sep="\t",
        label_sep=",",
        loader=None,
        extension: str = "",
        mask_extension: str = "",
        image_size: int = 256,
        microns_size: Tuple[int, int] = None,
        n_tiles: int = 36,
        tile_size: int = 256,
        tile_mode: int = 0,
        rand: bool = True,
        agg_type: str = "pazzle",
        use_coordinates: bool = False,
        use_adaptive_threshold: bool = False,
    ):
        self.img_root = img_root
        self.mask_root = mask_root
        self.transform = transform

        if use_crop_transform:
            self.crop_transform = Compose(
                [
                    Resize(height=image_size, width=image_size),
                    Transpose(),
                    VerticalFlip(),
                    HorizontalFlip(),
                ]
            )
        else:
            self.crop_transform = None

        self.loader = LOADER_MAPPING[loader] if loader else LOADER_MAPPING["cv2"]
        self.image_size = image_size
        self.microns_size = microns_size
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.tile_mode = tile_mode
        self.rand = rand
        self.agg_type = agg_type
        self.use_coordinates = use_coordinates
        self.use_adaptive_threshold = use_adaptive_threshold

        if label_mapping_path is None:
            label_mapping_path = os.path.join(self.img_root, LABEL_MAPPING_NAME)
        self.label_mapping: Dict[str, int] = load_label_mapping(label_mapping_path)
        self.label_column = label_column
        self.num_classes = len(self.label_mapping)
        self.index_file = pd.read_csv(file_name, sep=sep)
        self.label_sep = label_sep
        (
            self.paths,
            self.mask_paths,
            self.labels,
            self.data_providers,
            self.gleason_labels_first,
            self.gleason_labels_second,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for path, label, data_provider, gleason_score in self.index_file[
            [path_column, self.label_column, "data_provider", "gleason_score"]
        ].values:
            self.paths.append(os.path.join(self.img_root, path + extension))
            self.mask_paths.append(os.path.join(self.mask_root, path + mask_extension))
            self.labels.append(str(label))
            self.data_providers.append(data_provider)
            gleason_score = gleason_score.split("+")
            self.gleason_labels_first.append(gleason_score[0])
            self.gleason_labels_second.append(
                gleason_score[1] if len(gleason_score) > 1 else gleason_score[0]
            )

        self.target_transform = float_target_transform

    def read_img(self, index):
        image = self.loader(self.paths[index])
        return image

    def read_mask(self, index) -> Tuple[Optional[str], str]:
        path, data_provider = self.mask_paths[index], self.data_providers[index]
        if os.path.exists(path):
            return self.loader(path)[..., 0], data_provider
        else:
            return None, data_provider

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> dict:
        data_provider, tiles = self.prepare_tiles(index)

        if self.rand:
            idxes = np.random.choice(
                list(range(self.n_tiles)), self.n_tiles, replace=False
            )
        else:
            idxes = list(range(self.n_tiles))

        if self.agg_type == "pazzle":
            image, mask = self.make_pazzle(tiles, idxes)
        else:
            image, mask = self.make_pyramid(tiles, idxes)
        # mask = mask.unsqueeze(0)

        label = np.zeros(self.num_classes - 1).astype(np.float32)
        label[: self.get_mapping(self.labels[index])] = 1.0

        gleason_label_first = np.zeros(self.num_classes - 1).astype(np.float32)
        gleason_label_first[: self.get_mapping(self.gleason_labels_first[index])] = 1.0

        gleason_label_second = np.zeros(self.num_classes - 1).astype(np.float32)
        gleason_label_second[
            : self.get_mapping(self.gleason_labels_second[index])
        ] = 1.0

        if self.target_transform is not None:
            label = self.target_transform(label)
            gleason_label_first = self.target_transform(gleason_label_first)
            gleason_label_second = self.target_transform(gleason_label_second)

        return {
            "image": image,
            "mask": mask,
            "target": label,
            "gleason_first": gleason_label_first,
            "gleason_second": gleason_label_second,
            "data_provider": data_provider,
            "index": index,
        }

    def make_pyramid(self, tiles, idxes):
        count_tiles = len(tiles)
        channels = 5 if self.use_coordinates else 3
        image = torch.zeros(
            (channels, self.n_tiles, self.image_size, self.image_size),
            dtype=torch.float,
        )
        mask = torch.zeros(
            (self.n_tiles, self.image_size, self.image_size), dtype=torch.long,
        )
        for i in range(self.n_tiles):
            if count_tiles > idxes[i]:
                tile = tiles[idxes[i]]
                this_img, this_mask = tile.img, tile.mask

                if self.transform is not None:
                    if self.use_coordinates:
                        this_img, this_coords = np.split(this_img, [3], axis=-1)
                    transformed = self.transform(image=this_img, mask=this_mask)
                    this_img = transformed["image"]
                    this_mask = transformed["mask"]
                    if self.use_coordinates:
                        this_coords = torch.tensor(
                            np.moveaxis(this_coords, -1, 0), dtype=torch.float
                        )
                        this_img = torch.cat([this_img, this_coords], dim=0)

                image[:, i] = this_img
                mask[i] = this_mask
        return image, mask

    def make_pazzle(self, tiles, idxes):
        count_tiles = len(tiles)
        n_row_tiles = int(np.sqrt(self.n_tiles))
        channels = 5 if self.use_coordinates else 3
        image = np.zeros(
            (self.image_size * n_row_tiles, self.image_size * n_row_tiles, channels,),
            dtype=np.uint8,
        )
        mask = np.zeros(
            (self.image_size * n_row_tiles, self.image_size * n_row_tiles),
            dtype=np.long,
        )
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w

                if count_tiles > idxes[i]:
                    tile = tiles[idxes[i]]
                    this_img, this_mask = tile.img, tile.mask
                else:
                    this_img = (
                        np.ones((self.image_size, self.image_size, channels)).astype(
                            np.uint8
                        )
                        * 255
                    )
                    this_mask = np.zeros((self.image_size, self.image_size)).astype(
                        np.uint8
                    )

                if self.crop_transform is not None:
                    if self.use_coordinates:
                        this_img, this_coords = np.split(this_img, [3], axis=-1)
                    transformed = self.crop_transform(image=this_img, mask=this_mask)
                    this_img = transformed["image"]
                    this_mask = transformed["mask"]
                    if self.use_coordinates:
                        this_img = np.concatenate([this_img, this_coords], axis=-1)

                h1 = h * self.image_size
                w1 = w * self.image_size
                image[h1 : h1 + self.image_size, w1 : w1 + self.image_size] = this_img
                mask[h1 : h1 + self.image_size, w1 : w1 + self.image_size] = this_mask

        if self.transform is not None:
            if self.use_coordinates:
                image, coords = np.split(image, [3], axis=-1)
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            if self.use_coordinates:
                if isinstance(image, torch.Tensor):
                    coords = torch.tensor(np.moveaxis(coords, -1, 0), dtype=torch.float)
                    image = torch.cat([image, coords], dim=0)
                else:
                    image = np.concatenate([image, coords], axis=-1)
        return image, mask

    def prepare_tiles(self, index: int) -> Tuple[str, List[Tile]]:
        image = self.read_img(index)
        mask, data_provider = self.read_mask(index)
        if mask is None:
            mask = np.zeros_like(image)[..., 0]

        if self.microns_size is not None:
            pixels_shape = SpaceConverter(
                cm_resolution=OpenSlide(self.paths[index]).properties[
                    "tiff.XResolution"
                ]
            ).microns_to_pixels(microns_size=self.microns_size)
            self.tile_size = pixels_shape[0]

        tiles = self.get_tiles(image, mask, self.tile_mode)
        tiles_dataclass = []
        for i, tile in enumerate(tiles):
            image = tile["img"]
            mask = tile["mask"]
            tiles_dataclass.append(Tile(image, mask, i))

        return data_provider, tiles_dataclass

    def create_one_hot_mask(self, mask: NUMPY_TYPE, data_provider: str):
        if data_provider == "radboud":
            stroma_epithelium = ((1 <= mask) & (mask <= 2)).astype(np.uint8)
            cancerous = (mask > 2).astype(np.uint8)

        else:  # karolinska
            stroma_epithelium = (mask == 1).astype(np.uint8)
            cancerous = (mask == 2).astype(np.uint8)

        mask = np.stack([stroma_epithelium, cancerous], -1)
        return mask

    def get_mapping(self, x: str) -> int:
        return self.label_mapping.get(x, 0)

    def get_tiles(
        self, img, mask, mode=0,
    ):
        result = []
        h, w, c = img.shape
        pad_h = (self.tile_size - h % self.tile_size) % self.tile_size + (
            (self.tile_size * mode) // 2
        )
        pad_w = (self.tile_size - w % self.tile_size) % self.tile_size + (
            (self.tile_size * mode) // 2
        )

        img = np.pad(
            img,
            [
                [pad_h // 2, pad_h - pad_h // 2],
                [pad_w // 2, pad_w - pad_w // 2],
                [0, 0],
            ],
            constant_values=255,
        )

        mask = np.pad(
            mask,
            [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2]],
            constant_values=0,
        )

        if self.use_adaptive_threshold:
            img_binary = cv2.adaptiveThreshold(
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )

        if self.use_coordinates:
            img = self.add_coordinates(img)
            c += 2

        n_tiles_h = img.shape[0] // self.tile_size
        n_tiles_w = img.shape[1] // self.tile_size

        img = img.reshape((n_tiles_h, self.tile_size, n_tiles_w, self.tile_size, c))
        mask = mask.reshape((n_tiles_h, self.tile_size, n_tiles_w, self.tile_size))

        img = img.transpose(0, 2, 1, 3, 4).reshape(
            -1, self.tile_size, self.tile_size, c
        )
        mask = mask.transpose(0, 2, 1, 3).reshape(-1, self.tile_size, self.tile_size)

        if self.use_adaptive_threshold:
            img_binary = (
                img_binary.reshape(
                    (n_tiles_h, self.tile_size, n_tiles_w, self.tile_size)
                )
                .transpose(0, 2, 1, 3)
                .reshape(-1, self.tile_size, self.tile_size)
            )

        count_imgs = len(img)
        if count_imgs < self.n_tiles:
            img = np.pad(
                img,
                [[0, self.n_tiles - count_imgs], [0, 0], [0, 0], [0, 0]],
                constant_values=255,
            )
            mask = np.pad(
                mask,
                [[0, self.n_tiles - count_imgs], [0, 0], [0, 0]],
                constant_values=0,
            )
            if self.use_adaptive_threshold:
                img_binary = np.pad(
                    img_binary,
                    [[0, self.n_tiles - count_imgs], [0, 0], [0, 0]],
                    constant_values=0,
                )

        if self.use_adaptive_threshold:
            idxs = np.argsort(img_binary.reshape(img_binary.shape[0], -1).sum(-1))[
                ::-1
            ][: self.n_tiles]
        else:
            idxs = np.argsort(img[..., :3].reshape(img.shape[0], -1).sum(-1))[
                : self.n_tiles
            ]
        img = img[idxs]
        mask = mask[idxs]
        for i in range(self.n_tiles):
            result.append({"img": img[i], "mask": mask[i], "idx": i})
        return result

    @staticmethod
    def add_coordinates(image):
        image_height, image_width, channels = image.shape
        y_coords = (
            np.tile(2.0 * np.expand_dims(np.arange(image_height), 1), (1, image_width))
            / (image_height - 1)
        ) - 1
        x_coords = (
            np.tile(2.0 * np.expand_dims(np.arange(image_width), 0), (image_height, 1))
            / (image_width - 1)
        ) - 1
        coords = np.moveaxis(np.stack([y_coords, x_coords], axis=0), 0, -1)
        return np.concatenate([image, coords], -1)

    def get_tiles_with_intersection(self, img, mask, mode=0):
        result = []
        h, w, c = img.shape
        pad_h = (self.tile_size - h % self.tile_size) % self.tile_size + (
            (self.tile_size * mode) // 2
        )
        pad_w = (self.tile_size - w % self.tile_size) % self.tile_size + (
            (self.tile_size * mode) // 2
        )

        img = np.pad(
            img,
            [
                [pad_h // 2, pad_h - pad_h // 2],
                [pad_w // 2, pad_w - pad_w // 2],
                [0, 0],
            ],
            constant_values=255,
        )

        mask = np.pad(
            mask,
            [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2]],
            constant_values=0,
        )

        n_tiles_h = img.shape[0] // self.tile_size
        n_tiles_w = img.shape[1] // self.tile_size

        img = img.reshape((n_tiles_h, self.tile_size, n_tiles_w, self.tile_size, 3))
        mask = mask.reshape((n_tiles_h, self.tile_size, n_tiles_w, self.tile_size))

        img = img.transpose(0, 2, 1, 3, 4).reshape(
            -1, self.tile_size, self.tile_size, 3
        )
        mask = mask.transpose(0, 2, 1, 3).reshape(-1, self.tile_size, self.tile_size)

        count_imgs = len(img)
        if count_imgs < self.n_tiles:
            img = np.pad(
                img,
                [[0, self.n_tiles - count_imgs], [0, 0], [0, 0], [0, 0]],
                constant_values=255,
            )
            mask = np.pad(
                mask,
                [[0, self.n_tiles - count_imgs], [0, 0], [0, 0]],
                constant_values=0,
            )

        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[: self.n_tiles]
        img = img[idxs]
        mask = mask[idxs]
        for i in range(self.n_tiles):
            result.append({"img": img[i], "mask": mask[i], "idx": i})
        return result


@DATASETS.register_module
class ProstateCancerPreparedTilesDataset(ProstateCancerTileDataset):
    def read_img_tiles(self, index: int) -> List[NUMPY_TYPE]:
        images = []
        for i in range(self.n_tiles):
            images.append(self.loader(os.path.join(self.paths[index], f"{i}.png")))
        return images

    def read_mask_tiles(self, index: int) -> Tuple[List[NUMPY_TYPE], str]:
        images = []
        data_provider = self.data_providers[index]
        for i in range(self.n_tiles):
            img = self.loader(os.path.join(self.mask_paths[index], f"{i}.png"))[..., 0]
            images.append(img)
        return images, data_provider

    def prepare_tiles(self, index: int) -> Tuple[str, List[Tile]]:
        img_tiles = self.read_img_tiles(index)
        mask_tiles, data_provider = self.read_mask_tiles(index)
        tiles = []
        for i, (img_tile, mask_tile) in enumerate(zip(img_tiles, mask_tiles)):
            tiles.append(Tile(idx=i, img=img_tile, mask=mask_tile))
        return data_provider, tiles


class ProstateCancerTileDatasetInference(Dataset):
    def __init__(
        self,
        file_name: str,
        path_column: str = "image_id",
        img_root: str = "./",
        crop_transform=None,
        transform=None,
        sep="\t",
        loader=None,
        extension: str = "",
        image_size: int = 256,
        n_tiles: int = 36,
        tile_size: int = 256,
        tile_mode: int = 0,
        rand: bool = True,
    ):
        self.img_root = img_root
        self.transform = transform
        self.crop_transform = (
            crop_transform
            if crop_transform is not None
            else Compose(
                [
                    Resize(height=image_size, width=image_size),
                    Transpose(),
                    VerticalFlip(),
                    HorizontalFlip(),
                ]
            )
        )
        self.loader = LOADER_MAPPING[loader] if loader else LOADER_MAPPING["cv2"]
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.tile_mode = tile_mode
        self.rand = rand

        self.index_file = pd.read_csv(file_name, sep=sep)
        self.paths, self.data_providers = [], []
        for path, data_provider in zip(
            self.index_file[path_column], self.index_file["data_provider"],
        ):
            img_path = os.path.join(self.img_root, path + extension)
            self.paths.append(img_path)
            self.data_providers.append(data_provider)

    def read_img(self, index):
        path = self.paths[index]
        image = self.loader(path)
        return image, path.split("/")[-1].split(".")[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> dict:
        image, image_name = self.read_img(index)

        tiles = self.get_tiles(image, self.tile_mode)
        count_tiles = len(tiles)

        if self.rand:
            idxes = np.random.choice(
                list(range(self.n_tiles)), self.n_tiles, replace=False
            )
        else:
            idxes = list(range(self.n_tiles))

        n_row_tiles = int(np.sqrt(self.n_tiles))
        image = np.zeros(
            (self.image_size * n_row_tiles, self.image_size * n_row_tiles, 3),
            dtype=np.uint8,
        )
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w

                if count_tiles > idxes[i]:
                    tile = tiles[idxes[i]]
                    this_img = tile["img"]
                else:
                    this_img = (
                        np.ones((self.image_size, self.image_size, 3)).astype(np.uint8)
                        * 255
                    )

                if self.crop_transform is not None:
                    transform = self.crop_transform(image=this_img)
                    this_img = transform["image"]

                h1 = h * self.image_size
                w1 = w * self.image_size
                image[h1 : h1 + self.image_size, w1 : w1 + self.image_size] = this_img

        if self.transform is not None:
            transform = self.transform(image=image)
            image = transform["image"]

        return {
            "image": image,
            "index": index,
            "image_name": image_name,
        }

    def get_tiles(self, img, mode=0):
        result = []
        h, w, c = img.shape
        pad_h = (self.tile_size - h % self.tile_size) % self.tile_size + (
            (self.tile_size * mode) // 2
        )
        pad_w = (self.tile_size - w % self.tile_size) % self.tile_size + (
            (self.tile_size * mode) // 2
        )

        img = np.pad(
            img,
            [
                [pad_h // 2, pad_h - pad_h // 2],
                [pad_w // 2, pad_w - pad_w // 2],
                [0, 0],
            ],
            constant_values=255,
        )

        n_tiles_h = img.shape[0] // self.tile_size
        n_tiles_w = img.shape[1] // self.tile_size

        img = img.reshape((n_tiles_h, self.tile_size, n_tiles_w, self.tile_size, 3))

        img = img.transpose(0, 2, 1, 3, 4).reshape(
            -1, self.tile_size, self.tile_size, 3
        )

        count_imgs = len(img)
        if count_imgs < self.n_tiles:
            img = np.pad(
                img,
                [[0, self.n_tiles - count_imgs], [0, 0], [0, 0], [0, 0]],
                constant_values=255,
            )

        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[: self.n_tiles]
        img = img[idxs]
        for i in range(self.n_tiles):
            result.append({"img": img[i], "idx": i})
        return result