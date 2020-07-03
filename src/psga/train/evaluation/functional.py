from typing import Optional

import torch
from torch import Tensor

from .utils import nan_to_num


def confusion_matrix(input: Tensor, target: Tensor,
                     labels: Optional[Tensor] = None,
                     sample_weight: Optional[Tensor] = None,
                     normalize: Optional[str] = None) -> Tensor:
    dim = input.dim()
    if dim > 1:
        raise ValueError(f"Expected 2 or more dimensions (got {dim})")

    if input.size(0) != target.size(0):
        raise ValueError(f"Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).")

    if labels is None:
        labels = torch.unique(torch.cat([input, target]), sorted=True)
    else:
        n_labels = labels.size(0)
        if n_labels == 0:
            raise ValueError("'labels' should contains at least one label.")
        elif target.size(0) == 0:
            return torch.zeros(size=(n_labels, n_labels), dtype=torch.int)
        elif all([l not in target for l in labels]):
            raise ValueError("At least one label specified must be in target")

    if sample_weight is None:
        sample_weight = torch.ones(target.shape[0], dtype=torch.int64)
    else:
        sample_weight = torch.as_tensor(sample_weight)

    assert input.size(0) == target.size(0) == sample_weight.size(0), "Input dimension must match"

    if normalize not in ["true", "pred", "all", None]:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")

    n_labels = labels.size(0)
    label_to_ind = {int(y): x for x, y in enumerate(labels)}
    input = torch.as_tensor([label_to_ind.get(int(x), n_labels + 1) for x in input])
    target = torch.as_tensor([label_to_ind.get(int(x), n_labels + 1) for x in target])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = torch.logical_and(input < n_labels, target < n_labels)
    input = input[ind]
    target = target[ind]
    # also eliminate weights of eliminated items
    sample_weight = sample_weight[ind]

    indices = torch.stack([target, input], dim=0)
    cm = torch.sparse.FloatTensor(indices, sample_weight).to_dense()

    if normalize == "true":
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif normalize == "all":
        cm = cm / cm.sum()
    cm = nan_to_num(cm)

    return cm


def cohen_kappa_score(input: Tensor, target: Tensor,
                      weights: Optional[str] = None,
                      labels=None, sample_weight=None) -> Tensor:










    return torch.Tensor()
