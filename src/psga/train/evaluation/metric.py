from typing import (
    Optional,
    NoReturn
)

from torch import Tensor

from .functional import cohen_kappa_score


class QuadraticWeightedKappa(object):

    def __init__(self, labels: Optional[Tensor] = None, sample_weight: Optional[Tensor] = None) -> NoReturn:
        self._labels = labels
        self._sample_weight = sample_weight

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return cohen_kappa_score(input, target, weights="quadratic",
                                 labels=self._labels, sample_weight=self._sample_weight)
