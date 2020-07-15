from typing import (
    NoReturn,
    Optional
)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from .functional import cohen_kappa_score


class CohenKappaLoss(nn.Module):

    def __init__(self, scale: float = 2, weights: Optional[str] = None,
                 labels: Optional[Tensor] = None,
                 sample_weights: Optional[Tensor] = None) -> NoReturn:
        super().__init__()
        self._weights = weights
        self._labels = labels
        self._sample_weights = sample_weights
        self._scale = scale

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = torch.argmax(F.softmax(inputs, dim=1), dim=1)
        cohen_kappa = cohen_kappa_score(inputs, targets)
        return -torch.log(torch.sigmoid(self._scale * cohen_kappa))
