import pickle
from typing import (
    Any,
    List,
    NoReturn,
)


def save_pickle(data: Any, path: str) -> NoReturn:
    with open(path, "wb") as file_stream:
        pickle.dump(data, file_stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as file_stream:
        data = pickle.load(file_stream)
    return data


def load_file(path: str) -> List[str]:
    with open(path) as file_stream:
        lines = file_stream.read().splitlines()
    return lines
