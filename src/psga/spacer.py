from typing import (
    Tuple,
    Callable,
    NoReturn,
    Union
)

from typing_extensions import Final

MICRONS_PER_CM: Final = 10000
SIZE_TYPE = Union[int, Tuple[int, int]]


class SpaceConverter(object):

    def __init__(self, cm_resolution: float) -> NoReturn:
        self._pixel_spacing = 1 / (float(cm_resolution) / MICRONS_PER_CM)

    def _apply(self, inputs: SIZE_TYPE, func: Callable[[SIZE_TYPE], SIZE_TYPE]) -> SIZE_TYPE:
        if isinstance(inputs, int):
            return func(inputs)
        else:
            return (func(inputs[0]), func(inputs[1]))

    def pixels_to_microns(self, pixels_size: SIZE_TYPE) -> SIZE_TYPE:
        to_microns = lambda x: round(self._pixel_spacing * (x - 1))
        return self._apply(pixels_size, to_microns)

    def microns_to_pixels(self, microns_size: SIZE_TYPE) -> SIZE_TYPE:
        to_pixels = lambda x: round(x / self._pixel_spacing + 1)
        return self._apply(microns_size, to_pixels)
