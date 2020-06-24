import io
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
from typing_extensions import Final

from .base import (
    BaseDataAdapter,
    BaseWriter
)
from .record import Record
from .utils import draw_overlay_mask
from .phase import Phase
from ..image.preprocessing import NoneWhiteROICropper
from ..utils import load_pickle


RGB_TYPE = Tuple[Union[int, float], Union[int, float], Union[int, float]]


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

    def __init__(self, path: str, writer: BaseWriter, verbose: bool = False,
                 layer: int = 0, crop_nonwhite_roi: bool = True) -> NoReturn:
        super().__init__(path, writer, verbose)
        assert 0 <=layer <= 2, "Layers 0, 1, 2 are available"
        self._layer = layer
        self._crop_nonwhite_roi = crop_nonwhite_roi
        self._mp_namespace = Manager().Namespace()
        self._mp_namespace.self = self

    def convert(self, processes: int = 1) -> NoReturn:
        # to_paths = lambda path: sorted(path.rglob("*"))
        # image_paths = to_paths(self._path / "train_images")
        # mask_paths = to_paths(self._path / "train_label_masks")
        # mask_path_map = {x.stem.replace("_mask", ""): x for x in mask_paths}
        # paths = [(x, mask_path_map.get(x.stem, None)) for x in image_paths]
        paths = load_pickle("/data/processed/paths.pkl")

        train_meta = pd.read_csv(self._path / "train.csv")
        self._mp_namespace.train_meta = train_meta

        with Pool(processes) as p:
            run = lambda x: list(tqdm(x, total=len(paths), desc="Converted: ")) if self._verbose else lambda x: x
            run(p.imap(partial(self._worker, namespace=self._mp_namespace), paths))

        # from tqdm import tqdm
        # for path in tqdm(paths, total=len(paths)):
        #     self._worker(path, train_meta)

        print("flush")
        # self._writer.flush(count_samples_from="images/*/*")

    @staticmethod
    def _worker(paths: Tuple[Path, Optional[Path]], namespace) -> NoReturn:
        self = namespace.self
        train_meta = namespace.train_meta
        # train_meta = namespace
        image_path, mask_path = paths
        name = image_path.stem
        mask_path = Path(str(image_path).replace("train_images", "train_label_masks").replace(".tiff", "_mask.tiff"))

        image = self._read_image_safely(image_path)
        if image is None:
            return
        mask = self._read_mask_safely(mask_path) if mask_path.exists() else None
        if self._crop_nonwhite_roi:
            cropper = NoneWhiteROICropper()
            image, mask = cropper(image, mask)

        if 0 in image.shape:
            # Сделать выше, типа проверка и все пропускать, не менять функцию
            print(f"CONTINUE {name} with shape {image.shape}")
            return

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
            masked = draw_overlay_mask(image, mask, color_map=get_color_map(data_provider, normalized=False))
            title_text = f"{data_provider} - id={name[:10]} isup={label} gleason={gleason_score}"
            eda = self._to_eda(masked, title_text,
                               color_map=get_color_map(data_provider, normalized=True),
                               classname_map=get_classname_map(data_provider),
                               show_keys=list(np.unique(mask)))

        record = Record(image, mask, eda, name, label, phase=Phase.TRAIN, additional=additional)
        self._writer.put(record)

    def _read_image_safely(self, path: Path) -> Optional[np.ndarray]:
        try:
            return MultiImage(str(path))[self._layer]
        except Exception as e:
            print(f"{path.stem}[{self._layer}] - {e}")
        return None

    def _read_mask_safely(self, path: Path, across_layer_scale: int = 4) -> Optional[np.ndarray]:
        for current_layer in range(self._layer, 3):
            try:
                mask = MultiImage(str(path))[current_layer][..., 0]
                if current_layer != self._layer:
                    scale = across_layer_scale ** (current_layer - self._layer)
                    mask = cv2.resize(mask, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                return mask
            except Exception as e:
                print(f"{path.stem}[{current_layer}] - {e}")
        return None

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
