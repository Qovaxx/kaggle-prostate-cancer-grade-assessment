import albumentations as A
from albumentations import pytorch
from typing import (
    List,
    TypeVar
)

from ppln.utils.misc import object_from_dict
from ppln.utils.config import ConfigDict

T = TypeVar("T")


def make_albumentations(transforms: List[ConfigDict]) -> T:
    """
    Build transformation from albumentations library.
    Please, visit `https://albumentations.readthedocs.io` to get more information.

    Args:
        transforms (list): list of transformations to compose.
    """
    def build(config: ConfigDict) -> T:
        if "transforms" in config:
            config["transforms"] = [build(transform) for transform in config["transforms"]]
        try:
            return object_from_dict(config, A)
        except AttributeError:
            try:
                return object_from_dict(config, pytorch)
            except AttributeError:
                return object_from_dict(config)

    return build({"type": "Compose", "transforms": transforms})
