from typing import Tuple
import torch
from torch import Tensor


def accuracy(inputs: Tensor, targets: Tensor) -> Tensor:
    """basic accuracy based on equality.

    Args:
        inputs: binary values or class indices
        targets: ground truth binary values or class indices

    Returns:
        scalar tensor
    """
    return ((inputs == targets) * 1.0).mean()


def _onehot_inputs_and_targets(inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
    expanded_targets = targets.unsqueeze(-1)
    fill = inputs.new_ones(expanded_targets.shape)
    pred = torch.zeros_like(inputs).scatter_add(-1, inputs.argmax(dim=-1, keepdim=True), fill)
    tru = torch.zeros_like(inputs).scatter_add(-1, expanded_targets, fill)
    return pred, tru


def precision(inputs: Tensor, targets: Tensor, eps=1e-8, dim=0):
    tp = inputs * targets
    return (tp.sum(dim=dim) + eps) / (inputs.sum(dim=dim) + eps)


def recall(inputs: Tensor, targets: Tensor, eps=1e-8, dim=0):
    tp = inputs * targets
    return (tp.sum(dim=dim) + eps) / (targets.sum(dim=dim) + eps)


def fbeta(inputs: Tensor, targets: Tensor, beta: float, dim=0, eps=1e-8):
    r = recall(inputs, targets, dim=dim, eps=eps)
    p = precision(inputs, targets, dim=dim, eps=eps)
    return ((1 + beta ** 2) * p * r) / (beta ** 2 * p + r)


def f1(inputs: Tensor, targets: Tensor, dim=0, eps=1e-8):
    return fbeta(inputs, targets, beta=1.0, dim=dim, eps=eps)


def pearson_corr(inp: Tensor, target: Tensor) -> Tensor:
    return torch.corrcoef(torch.stack([inp, target])).amin()
