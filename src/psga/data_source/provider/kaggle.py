from pathlib import Path
from typing import (
    Dict,
    NoReturn,
    Optional,
    Tuple,
)
from functools import partial
from multiprocessing import (
    Pool,
    Manager,
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
from ...image_processing import ImagePreProcessor
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
        self._mp_namespace = Manager().Namespace()
        self._mp_namespace.self = self

    def convert(self, processes: int = 1) -> NoReturn:
        to_paths = lambda path: sorted(path.rglob("*"))
        image_paths = to_paths(self._path / "train_images")
        mask_paths = to_paths(self._path / "train_label_masks")
        mask_path_map = {x.stem.replace("_mask", ""): x for x in mask_paths}
        pairs = [(x, mask_path_map.get(x.stem, None)) for x in image_paths]

        train_meta = pd.read_csv(self._path / "train.csv")
        self._mp_namespace.train_meta = train_meta

        with Pool(processes) as p:
            run = lambda x: list(tqdm(x, total=len(pairs), desc="Converted: ")) if self._verbose else lambda x: x
            run(p.imap(partial(self._worker, namespace=self._mp_namespace), pairs))

        self._writer.flush(count_samples_from="images/*/*")

    @staticmethod
    def _worker(paths: Tuple[Path, Optional[Path]], namespace) -> NoReturn:
        self = namespace.self
        train_meta = namespace.train_meta
        image_path, mask_path = paths
        name = image_path.stem
        mask_path = Path(str(image_path).replace("train_images", "train_label_masks").replace(".tiff", "_mask.tiff"))

        image_slide = MultiImage(str(image_path))
        mask_slide = MultiImage(str(mask_path))
        large_image = get_layer_safely(image_slide, layer=0)
        large_mask = get_layer_safely(mask_slide, layer=0, is_mask=True) if mask_path.exists() else None
        small_image = get_layer_safely(image_slide, layer=2)
        if large_image is None:
            return

        if small_image is None:
            scale = 1 / 16
            small_image = cv2.resize(large_image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        try:
            pre_processor = ImagePreProcessor(reduce_memory=False)
            large_image = pre_processor.dual(large_image, small_image)
            if large_mask is not None:
                large_mask = pre_processor.single(large_mask)

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

            if large_mask is None:
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
