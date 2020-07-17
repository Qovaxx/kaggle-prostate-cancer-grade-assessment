from typing import (
    Optional,
    NoReturn
)

from torch import Tensor

from .functional import cohen_kappa_score


class Accuracy(object):

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(inputs.device)
        correct = (inputs == targets).float().sum()
        return correct / targets.size(0)


class QuadraticWeightedKappa(object):

    def __init__(self, labels: Optional[Tensor] = None, sample_weight: Optional[Tensor] = None) -> NoReturn:
        self._labels = labels
        self._sample_weight = sample_weight

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return cohen_kappa_score(inputs, targets, weights="quadratic",
                                 labels=self._labels, sample_weight=self._sample_weight)
