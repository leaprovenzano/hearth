from abc import ABC
from typing import Tuple, Mapping
import torch
from torch import Tensor
from hearth._internals import to_snakecase
from hearth.containers import TensorDict
from hearth._multihead import _MultiHeadFunc


class Metric(ABC):
    """abstract base class for all metrics.

    Note:
        Metrics should inherit from this method and define a forward method (for compatability
        with torch losses and modules)
    """

    def forward(self, inputs: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """given call this metric given an input and target and optional keyword arguments.

        Args:
            inp: the input tensor... generally some form of prediciton.
            target: the target tensor.

        Returns:
            a scalar tensor
        """
        return NotImplemented

    def _mask(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        return inputs, targets

    def _prepare(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        return inputs, targets

    def _aggregate(self, result: Tensor):
        return result

    def __call__(self, inp: Tensor, targets: Tensor, **kwargs) -> Tensor:
        with torch.no_grad():
            return self._aggregate(
                self.forward(*self._prepare(*self._mask(inp, targets)), **kwargs)
            )


class MetricStack(Metric):
    """a metric stack is a keyed collection of metric functions that will are all to be called with\
    a given set of inputs.

    This is useful when you'd like to run more than 1 metric function on an output

    Note:
        if you're using :class:`hearth.loop.Loop` and pass a list of metrics a metric stack
        will be created for you automatically.

    Example:
        >>> import torch
        >>> from hearth.metrics import BinaryAccuracy, BinaryF1, MetricStack
        >>> _ = torch.manual_seed(0)
        >>>
        >>> metrics = MetricStack(BinaryAccuracy(), BinaryF1())
        >>> metrics
        MetricStack(BinaryAccuracy(mask_target=-1, from_logits=False),
                    BinaryF1(eps=1e-08, dim=0, mask_target=-1, from_logits=False))

        if metrics are provided as args keys will be created based on the metric names
        >>> list(metrics.keys())
        ['binary_accuracy', 'binary_f1']


        you can access the individual metric functions by key if you need to:

        >>> metrics.binary_accuracy
        BinaryAccuracy(mask_target=-1, from_logits=False)

        when calling the metric stack with inputs and targets will compute all metrics
        for those inputs and targets and return a keyed :class:`TensorDict`.

        >>> inputs = torch.rand(10, 1)
        >>> targets = torch.rand(10, 1).round()
        >>>
        >>> metrics(inputs, targets)
        TensorDict({'binary_accuracy': tensor(0.7000), 'binary_f1': tensor(0.5714)})


        if you would rather choose your own keys you can instantiate the MetricStack
        with keyword args like so:

        >>> metrics = MetricStack(acc=BinaryAccuracy(), f1=BinaryF1())
        >>> metrics(inputs, targets)
        TensorDict({'acc': tensor(0.7000), 'f1': tensor(0.5714)})


        keyword arguments will be passed to all metric functions
        this is useful if you need them for some metrics and not others
        particularly because all hearth metrics accept variable keyword args
        and ignore them.

        >>> def my_metric(inputs, targets, weights, **kwargs):
        ...     return (inputs * weights).sum() / (targets * weights).sum()
        >>>
        >>> metrics = MetricStack(BinaryAccuracy(), my_metric)
        >>> weights = torch.normal(0, 5, size=(10,1))
        >>> metrics(inputs, targets, weights=weights)
        TensorDict({'binary_accuracy': tensor(0.7000), 'my_metric': tensor(23.2894)})
    """

    def __init__(self, *args, **kwargs):
        self._fns = {}
        for arg in args:
            self._add_fn(arg)
        for k, v in kwargs.items():
            self._add_fn(v, k)

    def _add_fn(self, fn, key=None):
        if not callable(fn):
            raise TypeError(f'functions passed to {self.__class__.__name__} must be callable!')
        if key is None:
            key = fn.__name__ if hasattr(fn, '__name__') else fn.__class__.__name__
            key = to_snakecase(key)
        self._fns[key] = fn

    def items(self):
        return self._fns.items()

    def keys(self):
        return self._fns.keys()

    def __getattr__(self, k):
        try:
            return self._fns[k]
        except KeyError:
            raise AttributeError(k)

    def __len__(self):
        return len(self._fns)

    def __call__(self, inputs, targets, **kwargs):
        return TensorDict({k: func(inputs, targets, **kwargs) for k, func in self.items()})

    def __repr__(self):
        _argrepr = ', '.join(f'{f!r}' for f in self._fns.values())
        return f'{self.__class__.__name__}({_argrepr})'


class MultiHeadMetric(_MultiHeadFunc):
    """a wrapper for metrics multi-output models.

    Example:
        >>> import torch
        >>> from hearth.metrics import MultiHeadMetric, BinaryAccuracy, CategoricalAccuracy
        >>> _ = torch.manual_seed(0)
        >>>
        >>> metric = MultiHeadMetric(a=BinaryAccuracy(), b=CategoricalAccuracy())
        >>> metric
        MultiHeadMetric(a=BinaryAccuracy(mask_target=-1, from_logits=False),
                        b=CategoricalAccuracy(mask_target=-1))


        inputs and targets should be some kind of mapping with head names matching
        those we specified in our metric.

        output will be a :class:`hearth.containers.TensorDict`

        >>> batch_size = 10
        >>> inputs = {'a': torch.rand(batch_size, 1),
        ...           'b': torch.normal(batch_size, 1, size=(10, 4))}
        >>> targets = {'a': torch.rand(batch_size, 1).round(),
        ...            'b': torch.randint(4, size=(batch_size,))}
        >>> metric(inputs, targets)
        TensorDict({'a': tensor(0.4000), 'b': tensor(0.6000)})
    """

    def __call__(
        self, inputs: Mapping[str, Tensor], targets: Mapping[str, Tensor], **kwargs
    ) -> TensorDict:
        out = TensorDict()
        for k, func in self.items():
            out[k] = func(inputs[k], targets[k], **kwargs)
        return out
