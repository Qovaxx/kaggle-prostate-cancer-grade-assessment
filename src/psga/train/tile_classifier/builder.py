from typing import TypeVar

import albumentations as A
from torch.utils.data import Dataset
from ppln.utils.misc import object_from_dict

from ..contrib.base import (
    _BaseBuilder,
    DPBaseBuilder,
    DDPBaseBuilder
)
from ..contrib.data.transforms import make_albumentations


T = TypeVar("T")


class TileClassifierBuilderMixin(_BaseBuilder):

    def transform(self, mode: str, level: str) -> A.Compose:
        return make_albumentations(self.config.TRANSFORMS[mode][level])

    def dataset(self, mode: str) -> Dataset:
        arguments = self.config.DATA[mode]
        arguments.update({"image_transforms": self.transform(mode, level="image_transforms")})
        arguments.update({"crop_transforms": self.transform(mode, level="crop_transforms")})
        return object_from_dict(arguments)


class TileClassifierDPBuilder(DPBaseBuilder, TileClassifierBuilderMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TileClassifierDDPBuilder(DDPBaseBuilder, TileClassifierBuilderMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
