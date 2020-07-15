from typing import (
    Optional,
    NoReturn
)

from torch import Tensor

from .functional import cohen_kappa_score


class Accuracy(object):

    def __init__(self, top_k: int = 1) -> NoReturn:
        self._top_k = top_k

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        batch_size = targets.size(0)
        _, pred = inputs.topk(self._top_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:self._top_k].view(-1).float().sum(0, keepdim=True)

        return correct_k.mul_(1.0 / batch_size)


class QuadraticWeightedKappa(object):

    def __init__(self, labels: Optional[Tensor] = None, sample_weight: Optional[Tensor] = None) -> NoReturn:
        self._labels = labels
        self._sample_weight = sample_weight

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return cohen_kappa_score(inputs, targets, weights="quadratic",
                                 labels=self._labels, sample_weight=self._sample_weight)
