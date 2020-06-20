import shutil
from abc import (
    abstractmethod,
    ABC
)
from typing import (
    Any,
    Callable,
    Dict,
    NoReturn,
)
from pathlib import Path

from .record import Record
from ..utils import (
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
    def attributes_path(self) -> Path:
        return self._root / "attributes.pkl"

    @property
    def images_path(self) -> Path:
        return self.data_path / "images"

    @property
    def masks_path(self) -> Path:
        return self.data_path / "masks"

    @property
    def eda_path(self) -> Path:
        return self.data_path / "eda"


class BaseWriter(BaseDataStructure):

    def __init__(self, path: str) -> NoReturn:
        super().__init__(path)
        self._dirs_exist = False
        self._attributes = list()
        self._temp_path = Path(self.root_path / "temp")

    @add_method_call("init_directories")
    def put(self, record: Record):
        tmp_pickle_path = (self._temp_path / record.name).with_suffix(".pkl")
        if not tmp_pickle_path.exists():
            attributes = self._put(record)
            save_pickle(attributes, path=str(tmp_pickle_path))

    @abstractmethod
    def _put(self, record: Record) -> Dict[str, Any]:
        ...

    def flush(self, path_template: str = "*/*") -> NoReturn:
        attributes = [load_pickle(str(path)) for path in self._temp_path.iterdir()]
        images_count = len(list(Path(self.data_path).rglob(path_template)))
        assert images_count == len(attributes), "Dimensions of data and attributes did not match"
        save_pickle(attributes, path=str(self.attributes_path))
        shutil.rmtree(self._temp_path)

    def init_directories(self) -> NoReturn:
        if not self._dirs_exist:
            self.data_path.mkdir(parents=True, exist_ok=True)
            self.images_path.mkdir(exist_ok=True)
            self.masks_path.mkdir(exist_ok=True)
            self.eda_path.mkdir(exist_ok=True)
            self._temp_path.mkdir(exist_ok=True)
            self._dirs_exist = True

    @staticmethod
    def _to_relative(path: Path) -> str:
        return str(path.relative_to(path.parent.parent.parent))


class BaseDataAdapter(ABC):

    def __init__(self, path: str, writer: BaseWriter, verbose: bool = False) -> NoReturn:
        self._path = Path(path)
        self._writer = writer
        self._verbose = verbose

    @abstractmethod
    def convert(self) -> NoReturn:
        ...
