import torch
from typing import Callable, Mapping, Union


class Running:
    """wrapper for metrics and losses for tracking running averages over batches.

    Args:
        fn: a loss or metric function.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> _ = torch.manual_seed(0)
        >>> from hearth.metrics import Running
        >>>
        >>> running_loss = Running(nn.BCELoss())
        >>> running_loss
        Running(BCELoss())

        >>> predictions = torch.rand(10, 1, requires_grad=True)
        >>> targets = (torch.rand(10, 1) > .5) *1.0
        >>> running_loss(predictions, targets) # this should have grad!
        tensor(0.5636, grad_fn=<BinaryCrossEntropyBackward0>)

        >>> running_loss.average
        0.5636

        generate some  new random inputs and run again this time with smaller batch:

        >>> predictions = torch.rand(6, 1, requires_grad=True)
        >>> targets = (torch.rand(6, 1) > .5) *1.0
        >>> running_loss(predictions, targets)
        tensor(1.0943, grad_fn=<BinaryCrossEntropyBackward0>)

        >>> running_loss.average
        0.7625999748706818

        call ``reset()`` to reset it:

        >>> running_loss.reset()
        >>> running_loss.average
        0.0

    """

    def __init__(self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.fn = fn
        self._reduction = getattr(fn, 'reduction', 'mean')
        self.reset()

    def reset(self):
        """reset all counters and totals on this metric."""
        self._batches_seen = 0
        self._samples_seen = 0
        self._total = 0

    @property
    def average(self) -> float:
        return self._total / max(self._samples_seen, 1)

    def _update(self, result, n_samples: int):
        if self._reduction == 'mean':
            self._total += result * n_samples
        elif self._reduction == 'sum':
            self._total += result
        self._samples_seen += n_samples
        self._batches_seen += 1

    def _get_n_samples(self, y):
        if isinstance(y, torch.Tensor):
            return y.shape[0]
        elif isinstance(y, Mapping):
            return self._get_n_samples(next(iter(y.values())))

    def __call__(self, inp, targets, **kwargs):
        res = self.fn(inp, targets, **kwargs)
        self._update(res.item(), self._get_n_samples(targets))
        return res

    def __repr__(self):
        return f'{self.__class__.__name__}({self.fn!r})'

    def to(self, device: Union[torch.device, str]):
        if hasattr(self.fn, 'to'):
            return self.fn.to(device)  # type: ignore
        return self
