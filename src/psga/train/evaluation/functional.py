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

    if normalize not in ["target", "input", "all", None]:
        raise ValueError("normalize must be one of {'target', 'input', 'all', None}")

    n_labels = labels.size(0)
    label_to_ind = {int(y): x for x, y in enumerate(labels)}
    input = torch.as_tensor([label_to_ind.get(int(x), n_labels + 1) for x in input])
    target = torch.as_tensor([label_to_ind.get(int(x), n_labels + 1) for x in target])

    # intersect input, target with labels, eliminate items not in labels
    ind = torch.logical_and(input < n_labels, target < n_labels)
    input = input[ind]
    target = target[ind]
    # also eliminate weights of eliminated items
    sample_weight = sample_weight[ind]

    indices = torch.stack([target, input], dim=0)
    cm = torch.sparse.FloatTensor(indices, sample_weight, torch.Size([n_labels, n_labels])).to_dense()

    if normalize == "target":
        cm = cm.float() / cm.sum(dim=1, keepdim=True)
    elif normalize == "input":
        cm = cm.float() / cm.sum(dim=0, keepdim=True)
    elif normalize == "all":
        cm = cm.float() / cm.sum()
    cm = nan_to_num(cm)

    return cm


def cohen_kappa_score(input: Tensor, target: Tensor,
                      weights: Optional[str] = None,
                      labels: Optional[Tensor] = None,
                      sample_weight: Optional[Tensor] = None) -> Tensor:
    """
    :param input: 1-dim vector
    :param target:  1-dim vector
    """
    cm_matrix = confusion_matrix(input, target, labels=labels, sample_weight=sample_weight)
    n_classes = cm_matrix.shape[0]
    sum0 = torch.sum(cm_matrix, dim=0)
    sum1 = torch.sum(cm_matrix, dim=1)
    expected_matrix = torch.ger(sum1, sum0).float() / torch.sum(sum1)

    if weights is None:
        weights_matrix = torch.ones((n_classes, n_classes), dtype=torch.int)
        weights_matrix.view(weights_matrix.numel())[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        weights_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int)
        weights_matrix += torch.arange(n_classes)
        if weights == "linear":
            weights_matrix = torch.abs(weights_matrix - weights_matrix.t())
        else:
            weights_matrix = (weights_matrix - weights_matrix.t()) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = torch.sum(weights_matrix * cm_matrix).float() / torch.sum(weights_matrix * expected_matrix)
    return 1 - k


def decode_ordinal_logits(input: torch.Tensor) -> torch.Tensor:
    classes = input.size(1)
    labels = torch.ones((input.size(0)), dtype=torch.long) * classes

    binned = input.sigmoid().round()
    zero_positions = (binned == 0).nonzero()
    row_indices_with_zero = zero_positions[:, 0]

    for index in torch.unique(row_indices_with_zero):
        label = zero_positions[(row_indices_with_zero == index).nonzero(), 1].min()
        labels[index] = label

    return labels


def decode_ordinal_labels(input: torch.Tensor) -> torch.Tensor:
    return input.sum(dim=1).long()
