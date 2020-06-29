from itertools import chain
from typing import (
    Optional, Tuple, List, NoReturn
)

import cv2
import rectpack
import numpy as np
from kornia import (
    image_to_tensor, tensor_to_image, warp_perspective, get_perspective_transform, get_rotation_matrix2d, warp_affine
)
from kornia.augmentation import RandomRotation
from kornia import Resample

from .types import (
    IMAGE_TYPE, MASK_TYPE, CV2_RECT_TYPE, RECTPACK_RECT_TYPE, SLICE_TYPE, Contour
)

from time import time
import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def validate_shape(image: IMAGE_TYPE, mask: MASK_TYPE) -> NoReturn:
    if mask is not None:
        assert image.shape[:2] == mask.shape, "Image and mask dimensions must match"


def crop_bbox(image: IMAGE_TYPE, bbox: np.ndarray) -> np.ndarray:
    bbox = np.int0(bbox)
    return image[bbox[0][1]: bbox[1][1], bbox[0][0]: bbox[1][0]]


def crop_slices(image: IMAGE_TYPE, row_slice: SLICE_TYPE, column_slice: SLICE_TYPE) -> np.ndarray:
    slice = image[row_slice, :]
    return slice[:, column_slice]


def scale_slice(slice: SLICE_TYPE, size: int) -> SLICE_TYPE:
    slice_image = np.asarray(slice).astype(np.uint8)
    slice_image = cv2.resize(slice_image, dsize=(1, size), interpolation=cv2.INTER_NEAREST)
    return slice_image.astype(np.bool).ravel().tolist()


def scale_rectangle(rectangle: CV2_RECT_TYPE, scale: int) -> CV2_RECT_TYPE:
    return (
        tuple(np.asarray(rectangle[0]) * scale),
        tuple(np.asarray(rectangle[1]) * scale),
        rectangle[2]
    )

def crop_rectangle(image: IMAGE_TYPE, rectangle: CV2_RECT_TYPE,
                   mask: MASK_TYPE = None) -> Tuple[IMAGE_TYPE, MASK_TYPE]:

    image = np.zeros((32000, 32000))


    def rotate_bound(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    start = time()
    zz = rotate_bound(image, abs(rectangle[2]))
    print(time() - start)
    # show(zz)



    a = 4











# TODO: REFACTOR

def crop_rectangle_old(image: IMAGE_TYPE, rectangle: CV2_RECT_TYPE,
                       mask: MASK_TYPE = None) -> Tuple[IMAGE_TYPE, MASK_TYPE]:

    box = image_to_tensor(cv2.boxPoints(box=rectangle))
    width = int(rectangle[1][0])
    height = int(rectangle[1][1])
    redl1ference_bbox = image_to_tensor(np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
                                              dtype=np.float32))
    image_tensor = image_to_tensor(image, keepdim=False)
    transformation_matrix = get_perspective_transform(src=box, dst=reference_bbox)

    print("warp_image")
    warped_image = warp_perspective(src=image_tensor.float(), M=transformation_matrix, dsize=(height, width),
                                    flags="bilinear", border_mode="border")
    warped_image = tensor_to_image(warped_image.byte())
    warped_mask = None
    if mask is not None:
        print("warp_mask")
        mask_tensor = image_to_tensor(mask, keepdim=False)
        warped_mask = warp_perspective(src=mask_tensor.float(), M=transformation_matrix, dsize=(height, width),
                                       flags="nearest", border_mode="zeros")
        warped_mask = tensor_to_image(warped_mask.byte())

    return warped_image, warped_mask


def apply_mask(image: np.ndarray, mask: np.ndarray, add: int = 0) -> np.ndarray:
    assert image.shape[:2] == mask.shape
    return cv2.bitwise_and(src1=image, src2=image, mask=mask) + add


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
        pairs.append(crop_rectangle_old(contoured_image, contour.rectangle, contoured_mask))

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
