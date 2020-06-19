import pickle
from abc import (
    abstractmethod,
    ABC
)
from typing import (
    Any,
    Callable,
    NoReturn,
)
from pathlib import Path

from .record import Record


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

    @add_method_call("init_directories")
    def put(self, *args, **kwargs) -> NoReturn:
        return self._put(*args, **kwargs)

    @abstractmethod
    def _put(self, record: Record) -> NoReturn:
        ...

    def flush(self, path_template: str = "*/*") -> NoReturn:
        images_count = len(list(Path(self.data_path).rglob(path_template)))
        assert images_count == len(self._attributes), "Dimensions of data and attributes did not match"
        self._save_pickle(data=self._attributes, path=str(self.attributes_path))

    def init_directories(self) -> NoReturn:
        if not self._dirs_exist:
            self.data_path.mkdir(parents=True, exist_ok=True)
            self.images_path.mkdir(exist_ok=True)
            self.masks_path.mkdir(exist_ok=True)
            self.eda_path.mkdir(exist_ok=True)
            self._dirs_exist = True

    @staticmethod
    def _save_pickle(data: Any, path: str) -> NoReturn:
        with open(path, "wb") as file_stream:
            pickle.dump(data, file_stream, protocol=pickle.HIGHEST_PROTOCOL)


class BaseDataAdapter(ABC):

    def __init__(self, path: str, writer: BaseWriter, verbose: bool = False) -> NoReturn:
        self._path = Path(path)
        self._writer = writer
        self._verbose = verbose

    @abstractmethod
    def convert(self) -> NoReturn:
        ...
