import logging

import pyvips
import numpy as np

logging.getLogger("pyvips").setLevel(logging.WARNING)

__all__ = ["from_numpy", "to_numpy", "imread"]


FORMAT_MAP = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
    "complex": np.complex64,
    "dpcomplex": np.complex128,
}

DTYPE_MAP = {
    "uint8": 'uchar',
    "int8": 'char',
    "uint16": 'ushort',
    "int16": 'short',
    "uint32": 'uint',
    "int32": 'int',
    "float32": 'float',
    "float64": 'double',
    "complex64": 'complex',
    "complex128": 'dpcomplex',
}


def from_numpy(numpy_image: np.ndarray) -> pyvips.Image:
    height, width = numpy_image.shape[:2]
    bands = numpy_image.shape[2] if len(numpy_image.shape) == 3 else 1
    flatten = numpy_image.reshape(width * height * bands)
    vips_image = pyvips.Image.new_from_memory(flatten.data, width, height, bands,
                                              format=DTYPE_MAP[str(numpy_image.dtype)])
    return vips_image


def to_numpy(vips_image: pyvips.Image) -> np.ndarray:
    return np.ndarray(buffer=vips_image.write_to_memory(),
                      dtype=FORMAT_MAP[vips_image.format],
                      shape=[vips_image.height, vips_image.width, vips_image.bands])


def imread(path: str, **kwargs) -> np.ndarray:
    vips_image = pyvips.Image.new_from_file(path, **kwargs)
    if vips_image.bands == 1:
        vips_image = vips_image.colourspace("b-w")

    return to_numpy(vips_image)
