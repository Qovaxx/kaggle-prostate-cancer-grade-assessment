import os
import io
from typing import (
    Dict,
    NoReturn,
    Optional,
    List,
    Tuple,
    Union
)
from tqdm import tqdm

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.io import MultiImage

from .base import (
    BaseDataAdapter,
    BaseWriter
)
from .record import Record
from .utils import (
    crop_tissue_roi,
    draw_overlay_mask
)
from .phase import Phase

RGB_TYPE = Tuple[Union[int, float], Union[int, float], Union[int, float]]

SEGMENTATION_MAP = {
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


class PSGADataAdapter(BaseDataAdapter):

    def __init__(self, path: str, writer: BaseWriter, verbose: bool = False,
                 layer: int = 0, crop_tissue_roi: bool = True) -> NoReturn:
        super().__init__(path, writer, verbose)
        self._layer = layer
        self._crop_tissue_roi = crop_tissue_roi

    def get_classname_map(self, data_provider: str) -> Dict[int, str]:
        return {k: v[0] for k, v in SEGMENTATION_MAP[data_provider].items()}

    def get_color_map(self, data_provider: str, normalized: bool = False) -> Dict[int, RGB_TYPE]:
        normalize_it = lambda x: tuple(np.asarray(x) / 255) if normalized else x
        return {k: normalize_it(v[1]) for k, v in SEGMENTATION_MAP[data_provider].items()}

    def convert(self) -> NoReturn:
        to_paths = lambda path: sorted([str(x) for x in path.rglob("*")])
        train_meta = pd.read_csv(self._path / "train.csv")
        image_paths = to_paths(self._path / "train_images")
        mask_paths = to_paths(self._path / "train_label_masks")

        iter = image_paths
        if self._verbose:
            iter = tqdm(iter, total=len(iter), desc="Converted: ")

        for image_path in iter:
            name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = image_path.replace("train_images", "train_label_masks").replace(".tiff", "_mask.tiff")

            image = MultiImage(image_path)[self._layer]
            mask = MultiImage(mask_path)[self._layer][..., 0] if mask_path in mask_paths else None
            if self._crop_tissue_roi:
                if mask is None:
                    image, _ = crop_tissue_roi(image)
                else:
                    image, additional_images = crop_tissue_roi(image, additional_images=[mask])
                    mask = additional_images[0]

            row = train_meta[train_meta.image_id == name].iloc[0]
            data_provider = row["data_provider"]
            gleason_score = row["gleason_score"]
            label = row["isup_grade"]
            additional = {"data_provider": data_provider,
                          "gleason_score": gleason_score,
                          "image_shape": image.shape[:2]}

            if mask is None:
                eda = None
            else:
                masked = draw_overlay_mask(image, mask, color_map=self.get_color_map(data_provider, normalized=False))
                title_text = f"{data_provider} - id={name[:10]} isup={label} gleason={gleason_score}"
                eda = self._to_eda(masked, title_text,
                                   color_map=self.get_color_map(data_provider, normalized=True),
                                   classname_map=self.get_classname_map(data_provider),
                                   show_keys=list(np.unique(mask)))

            record = Record(image, mask, eda, name, label, phase=Phase.TRAIN, additional=additional)
            self._writer.put(record)
        self._writer.flush(path_template="images/*/*")

    @staticmethod
    def _to_eda(image: np.ndarray, title_text: str,
                color_map: Dict[int, RGB_TYPE], classname_map: Dict[int, str],
                dpi: int = 300, show_keys: Optional[List[int]] = None) -> np.ndarray:

        assert len(set(color_map.keys()).difference(set(classname_map.keys()))) == 0,\
            "Color map and class name map must contain the same keys"
        keys = color_map.keys() if show_keys is None else show_keys
        patches = [Patch(color=color_map[x], label=classname_map[x]) for x in keys]

        plt.figure()
        plt.title(title_text)
        plt.legend(handles=patches, bbox_to_anchor=(1.04, 0.5), loc="center left", prop={"size": 6})
        plt.imshow(image)

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi)
        plt.close()
        buffer.seek(0)
        eda = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        buffer.close()

        return cv2.cvtColor(cv2.imdecode(eda, 1), cv2.COLOR_BGR2RGB)
