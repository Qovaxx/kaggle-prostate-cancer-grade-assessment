import pickle
from typing import (
    Any,
    NoReturn,
)


def save_pickle(data: Any, path: str) -> NoReturn:
    with open(path, "wb") as file_stream:
        pickle.dump(data, file_stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as file_stream:
        data = pickle.load(file_stream)
    return data
