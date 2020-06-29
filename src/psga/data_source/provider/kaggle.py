from pathlib import Path
from functools import partial
from multiprocessing import (
    Pool,
    Manager,
)
from typing import (
    Dict,
    NoReturn,
    Optional,
    Tuple,
    Union
)
from tqdm import tqdm

import cv2
import pandas as pd
import numpy as np
from openslide import OpenSlide
from skimage.io import MultiImage
from typing_extensions import Final

from ..base import (
    BaseDataAdapter,
    BaseWriter
)
from ..record import Record
from ..utils import (
    draw_overlay_mask,
    plot_meta,
    RGB_TYPE
)
from ...processing import (
    compose_preprocessing,
    dual_compose_preprocessing
)
from ...phase import Phase
from ...utils.slide import get_layer_safely


SEGMENTATION_MAP: Final = {
    "radboud": {
        0: ("background (non tissue) or unknown", (255, 255, 255)),
        1: ("stroma (connective tissue, non-epithelium tissue)", (0, 0, 255)),
        2: ("healthy (benign) epithelium", (0, 255, 0)),
        3: ("cancerous epithelium (Gleason 3)", (255, 255, 0)),
        4: ("cancerous epithelium (Gleason 4)", (255, 128, 0)),
        5: ("cancerous epithelium (Gleason 5)", (255, 0, 0))
    },
    "karolinska": {
        0: ("background (non tissue) or unknown", (255, 255, 255)),
        1: ("benign tissue (stroma and epithelium combined)", (0, 255, 0)),
        2: ("cancerous tissue (stroma and epithelium combined)", (255, 0, 0))
    }
}


def get_classname_map(data_provider: str) -> Dict[int, str]:
    return {k: v[0] for k, v in SEGMENTATION_MAP[data_provider].items()}


def get_color_map(data_provider: str, normalized: bool = False) -> Dict[int, RGB_TYPE]:
    normalize_it = lambda x: tuple(np.asarray(x) / 255) if normalized else x
    return {k: normalize_it(v[1]) for k, v in SEGMENTATION_MAP[data_provider].items()}


class PSGADataAdapter(BaseDataAdapter):

    def __init__(self, path: str, writer: BaseWriter, verbose: bool = False) -> NoReturn:
        super().__init__(path, writer, verbose)

    def convert(self) -> NoReturn:
        to_paths = lambda path: sorted(path.rglob("*"))
        image_paths = to_paths(self._path / "train_images")
        mask_paths = to_paths(self._path / "train_label_masks")
        mask_path_map = {x.stem.replace("_mask", ""): x for x in mask_paths}
        pairs = [(x, mask_path_map.get(x.stem, None)) for x in image_paths]
        train_meta = pd.read_csv(self._path / "train.csv")

        iter = pairs[2:]
        if self._verbose:
            iter = tqdm(iter, total=len(iter), desc="Converted: ")

        for paths in iter:
            self._worker(paths, train_meta)

        self._writer.flush(count_samples_from="images/*/*")

    def _worker(self, paths: Tuple[Path, Optional[Path]], train_meta: pd.DataFrame) -> NoReturn:
        image_path, mask_path = paths
        name = image_path.stem
        mask_path = Path(str(image_path).replace("train_images", "train_label_masks").replace(".tiff", "_mask.tiff"))

        image_slide = MultiImage(str(image_path))
        mask_slide = MultiImage(str(mask_path))
        large_image = get_layer_safely(image_slide, layer=0)
        large_mask = get_layer_safely(mask_slide, layer=0, is_mask=True)
        small_image = get_layer_safely(image_slide, layer=2)

        if large_image is None:
            return

        if small_image is None:
            scale = 1 / 16
            small_image = cv2.resize(large_image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        try:
            large_image, intermediates = dual_compose_preprocessing(large_image, small_image, reduce_memory=False)
            if large_mask is not None:
                large_mask, _ = compose_preprocessing(large_mask, intermediates=intermediates, reduce_memory=False)

            row = train_meta[train_meta.image_id == name].iloc[0]
            data_provider = row["data_provider"]
            gleason_score = row["gleason_score"]
            label = row["isup_grade"]
            slide = OpenSlide(str(image_path))
            additional = {"data_provider": data_provider,
                          "gleason_score": gleason_score,
                          "image_shape": large_image.shape[:2],
                          "source_image_shape": slide.dimensions,
                          "x_resolution": float(slide.properties["tiff.XResolution"]),
                          "y_resolution": float(slide.properties["tiff.YResolution"]),
                          "resolution_unit": slide.properties["tiff.ResolutionUnit"]}

            if large_image is None:
                visualization = None
            else:
                masked = draw_overlay_mask(large_image, large_mask, color_map=get_color_map(data_provider, normalized=False))
                title_text = f"{data_provider} - id={name[:10]} isup={label} gleason={gleason_score}"
                visualization = plot_meta(masked, title_text,
                                          color_map=get_color_map(data_provider, normalized=True),
                                          classname_map=get_classname_map(data_provider),
                                          show_keys=list(np.unique(large_mask)))

            record = Record(large_image, large_mask, visualization, name, label, phase=Phase.TRAIN,
                            additional=additional)
            self._writer.put(record)

        except Exception as e:
            print(f"{name} - {e}")
