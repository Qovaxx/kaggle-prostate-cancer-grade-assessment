from typing import (
    Optional,
    NoReturn
)

import albumentations as A
from torch.utils.data import Dataset

from ...read import TIFFReader
from ...base import BasePhaseSplitter
from ...split import CleanedPhaseSplitter
from ....phase import Phase



class PSGAClassificationDataset(Dataset):

    def __init__(self, path: str, fold: int = 0, phase: Phase = Phase.TRAIN,
                 transforms: Optional[A.Compose] = None,
                 phase_splitter: Optional[BasePhaseSplitter] = CleanedPhaseSplitter()) -> NoReturn:
        self._reader = TIFFReader(path)
        self._transforms = transforms
        if phase_splitter:
            phase_splitter.split(self._reader)


        a = 4


    # def __len__(self) -> int:
    #     return 1
    #
    # def __getitem__(self, item: int) -> Dict:
    #     return
