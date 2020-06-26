from itertools import chain
from typing import (
    Optional,
    Tuple,
    List
)

import cv2
import rectpack
import numpy as np
from kornia import (
    image_to_tensor,
    tensor_to_image,
    warp_perspective,
    get_perspective_transform
)


from .type import (
    Contour,
    CV2_RECT_TYPE,
    RECTPACK_RECT_TYPE
)


def apply_mask(image: np.ndarray, mask: np.ndarray, add: int = 0) -> np.ndarray:
    assert image.shape[:2] == mask.shape
    return cv2.bitwise_and(src1=image, src2=image, mask=mask) + add


def scale_rectangle(rectangle: CV2_RECT_TYPE, scale: int) -> CV2_RECT_TYPE:
    return (
        tuple(np.asarray(rectangle[0]) * scale),
        tuple(np.asarray(rectangle[1]) * scale),
        rectangle[2]
    )


def crop_bbox(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    bbox = np.int0(bbox)
    return image[bbox[0][1]: bbox[1][1], bbox[0][0]: bbox[1][0]]


def crop_rectangle(image: np.ndarray, rectangle: CV2_RECT_TYPE,
                   mask: Optional[np.ndarray] = None) -> Tuple[np.array, Optional[np.array]]:
    box = image_to_tensor(cv2.boxPoints(box=rectangle))
    width = int(rectangle[1][0])
    height = int(rectangle[1][1])
    reference_bbox = image_to_tensor(np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype=np.float32))
    image_tensor = image_to_tensor(image, keepdim=False)
    transformation_matrix = get_perspective_transform(src=box, dst=reference_bbox)

    print("warp_image")
    crop = warp_perspective(src=image_tensor.float(), M=transformation_matrix,
                                   dsize=(height, width), flags="bilinear", border_mode="border")
    crop = tensor_to_image(crop.byte())

    if mask is not None:
        print("warp_mask")
        mask_tensor = image_to_tensor(mask, keepdim=False)
        mask = warp_perspective(src=mask_tensor.float(), M=transformation_matrix,
                                dsize=(height, width), flags="nearest", border_mode="zeros")
        mask = tensor_to_image(mask.byte())

    return crop, mask


def pack_bin(contours: List[Contour], step_size: int = 10) -> Tuple[List[RECTPACK_RECT_TYPE], Tuple[int, int]]:
    sides = [(int(contour.rectangle[1][0]), int(contour.rectangle[1][1])) for contour in contours]
    sides = list(chain(*sides))
    height = max(sides)
    width = min(sides)

    while True:
        packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline,
                                    pack_algo=rectpack.MaxRectsBlsf,
                                    sort_algo=rectpack.SORT_LSIDE,
                                    rotation=True)
        packer.add_bin(width=width, height=height)
        for index, contour in enumerate(contours):
            packer.add_rect(width=int(contour.rectangle[1][1]), height=int(contour.rectangle[1][0]), rid=index)

        packer.pack()
        if len(contours) == len(packer[0].rectangles):
            break
        else:
            width += step_size

    return packer.rect_list(), (height, width)


def pack_atlas(image: np.ndarray, atlas_shape: Tuple[int, int],
               contours: List[Contour], rectangles: List[RECTPACK_RECT_TYPE],
               mask: Optional[np.ndarray] = None,
               background_value: int = 255) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert len(contours) == len(rectangles)
    if mask is not None:
        assert image.shape[:2] == mask.shape

    pairs = list()
    for contour in contours:
        contoured_image = apply_mask(image, mask=contour.mask, add=background_value)
        contoured_mask = apply_mask(mask, mask=contour.mask) if mask is not None else None
        pairs.append(crop_rectangle(contoured_image, contour.rectangle, contoured_mask))

    def __pack(atlas: np.ndarray, crops: List[np.ndarray]) -> np.ndarray:
        for rectangle in rectangles:
            _, x, y, w, h, index = rectangle
            crop = crops[index]
            if crop.shape[:2] != (h, w):
                crop = np.rot90(crop)
            atlas[y: y + h, x: x + w] = crop
        return atlas
    slice_list = lambda iter, axis: list(map(lambda x: x[axis], iter))

    image_atlas = __pack(atlas=np.full(shape=(*atlas_shape, 3), fill_value=255, dtype=np.uint8),
                         crops=slice_list(pairs, axis=0))
    mask_atlas = None
    if mask is not None:
        mask_atlas = __pack(atlas=np.zeros(shape=atlas_shape, dtype=np.uint8),
                            crops=slice_list(pairs, axis=1))

    return image_atlas, mask_atlas