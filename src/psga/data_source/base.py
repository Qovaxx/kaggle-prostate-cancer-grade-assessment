import shutil
from abc import (
    abstractmethod,
    ABC
)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    NoReturn,
)
from pathlib import Path

from .record import Record
from src.psga.utils.pickle import (
    save_pickle,
    load_pickle
)


def add_method_call(method_name: str) -> Callable:
    def decorator(function: Callable) -> Callable:
        def wrapper(self, *args, **kwargs) -> Any:
            getattr(self, method_name)()
            return function(self, *args, **kwargs)
        return wrapper
    return decorator


class BaseDataStructure(ABC):

    def __init__(self, path: str) -> NoReturn:
        self._root = Path(path)

    @property
    def root_path(self) -> Path:
        return self._root

    @property
    def data_path(self) -> Path:
        return self._root / "data"

    @property
    def meta_path(self) -> Path:
        return self._root / "meta.pkl"

    @property
    def images_path(self) -> Path:
        return self.data_path / "images"

    @property
    def masks_path(self) -> Path:
        return self.data_path / "masks"

    @property
    def visualizations_path(self) -> Path:
        return self.data_path / "visualizations"


class BaseWriter(BaseDataStructure):

    def __init__(self, path: str) -> NoReturn:
        super().__init__(path)
        self._dirs_exist = False
        self._meta: Optional[List[Dict[str, Any]]] = list()
        self._temp_path = Path(self.root_path / "temp")

    @add_method_call("init_directories")
    def put(self, record: Record):
        tmp_pickle_path = (self._temp_path / record.name).with_suffix(".pkl")
        if not tmp_pickle_path.exists():
            meta = self._put(record)
            save_pickle(meta, path=str(tmp_pickle_path))

    @abstractmethod
    def _put(self, record: Record) -> Dict[str, Any]:
        ...

    def flush(self, count_samples_from: str = "*/*") -> NoReturn:
        meta = [load_pickle(str(path)) for path in self._temp_path.iterdir()]
        images_count = len(list(Path(self.data_path).rglob(count_samples_from)))
        assert images_count == len(meta), "Dimensions of data_source and meta did not match"
        save_pickle(meta, path=str(self.meta_path))
        shutil.rmtree(self._temp_path)

    def init_directories(self) -> NoReturn:
        if not self._dirs_exist:
            self.data_path.mkdir(parents=True, exist_ok=True)
            self.images_path.mkdir(exist_ok=True)
            self.masks_path.mkdir(exist_ok=True)
            self.visualizations_path.mkdir(exist_ok=True)
            self._temp_path.mkdir(exist_ok=True)
            self._dirs_exist = True

    @staticmethod
    def _to_relative(path: Path) -> str:
        return str(path.relative_to(path.parent.parent.parent))


class BaseReader(BaseDataStructure):

    def __init__(self, path: str) -> NoReturn:
        super().__init__(path)
        assert self.data_path.exists(), "Directory with data_source not found"
        assert self.meta_path.exists(), "Samples meta file not found"
        # Preventing the copying of a large object between processes
        self.__meta: Optional[List[Dict[str, Any]]] = None

    @property
    def meta(self) -> List[Dict[str, Any]]:
        if self.__meta is None:
            self.__meta = load_pickle(str(self.meta_path))
        return self.__meta

    @abstractmethod
    def get(self, index: int) -> Record:
        ...

    @property
    def num_images(self) -> int:
        return len(self.meta)


class BaseDataAdapter(ABC):

    def __init__(self, path: str, writer: BaseWriter, verbose: bool = False) -> NoReturn:
        self._path = Path(path)
        self._writer = writer
        self._verbose = verbose

    @abstractmethod
    def convert(self) -> NoReturn:
        ...
