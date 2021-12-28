"""transforms are basic operations that can be composed as part of a `Pipeline`_
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Union, Generic

import torch
import numpy as np

InT = TypeVar('InT')
OutT = TypeVar('OutT')
TensorApplicable = Union[torch.Tensor, np.ndarray, int, float]


class Transform(ABC, Generic[InT, OutT]):
    """Abstract base class for all transforms."""

    def _repr_args(self):
        return ''

    def __repr__(self):
        return f'{self.__class__.__name__}({self._repr_args()})'

    @abstractmethod
    def __call__(self, x: InT) -> OutT:
        return NotImplemented


class Tensorize(Transform[InT, torch.Tensor]):
    """Tensorizes the given input with optional dtype and device

    Args:
        dtype : an optional string or torch.dtype. Defaults to None.
        device : [description]. Defaults to 'cpu'.

    Example:
        >>> import torch
        >>> from hearth.data.transforms import Tensorize
        >>>
        >>> transform = Tensorize(dtype='float32')
        >>> transform([1.1, 2.2, 3.3])
        tensor([1.1000, 2.2000, 3.3000])
    """

    def __init__(
        self,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Union[str, torch.device] = 'cpu',
    ):
        self._dtype = self._get_dtype(dtype)
        self._device = device

    def _get_dtype(self, dtype):
        if dtype is None:
            return dtype

        try:
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)
            return dtype
        except (AttributeError, AssertionError):
            raise TypeError(
                'dtype must be a valid torch.dtype or a string'
                f' corresponding to a valid torch.dtype got {dtype}.'
            )

    def _repr_args(self):
        return f'dtype={self._dtype}, device={self._device}'

    def __call__(self, x: InT) -> torch.Tensor:
        return torch.tensor(x, dtype=self._dtype, device=self._device)


class Normalize(Transform):
    """Normalize a tensor or array from a fixed mean and std

    Args:
        mean : may be float, tensor or array. if has dimensions (such as channels for images)
            must match shape of std
        std : may be float, tensor or array. if has dimensions (such as channels for images)
            must match shape of std

    Example:
        >>> import torch
        >>> from hearth.data.transforms import Normalize
        >>>
        >>> transform = Normalize(mean=1.5, std=1.1859)
        >>> x = torch.linspace(0, 3, 5)
        >>> transform(x)
        tensor([-1.2649, -0.6324,  0.0000,  0.6324,  1.2649])

        >>> channel_transform = Normalize(mean=torch.tensor([7.6596, 8.0000, 8.3404]),
        ...                                 std=torch.tensor([4.8622, 4.8622, 4.8622]))
        >>> x= torch.linspace(0, 16, 48).reshape(4, 4, 3)
        >>> y = channel_transform(x)
        >>> y.shape
        torch.Size([4, 4, 3])

        >>> y.mean(dim=(0, 1))
        tensor([-0.00,  0.00,  0.00])

        >>> y.std(dim=(0, 1))
        tensor([1.0000, 1.0000, 1.0000])
    """

    def __init__(self, mean: TensorApplicable, std: TensorApplicable):
        self.mean = mean
        self.std = std

    def _repr_args(self):
        return f'mean={self.mean}, std={self.std}'

    def __call__(self, x):
        return (x - self.mean) / self.std


class Pipeline(Transform):
    """Pipeline applies a chain of transforms to an input in order.

    Example:
        >>> import torch
        >>> import numpy as np
        >>> from hearth.data.transforms import Normalize, Tensorize, Pipeline
        >>>
        >>> pipeline = Pipeline(Tensorize(dtype='float32'), Normalize(mean=-0.34, std=1.75))
        >>> pipeline
        Pipeline(Tensorize(dtype=torch.float32, device=cpu), Normalize(mean=-0.34, std=1.75))

        >>> len(pipeline)
        2

        >>> x = np.array([-3.0, -1.5, .3, .4, 2.1])
        >>> pipeline(x)
        tensor([-1.5200, -0.6629,  0.3657,  0.4229,  1.3943])
    """

    def __init__(self, *transforms):
        self._transforms = transforms

    def _repr_args(self):
        return ', '.join(map('{!r}'.format, self._transforms))

    def __len__(self) -> int:
        return len(self._transforms)

    def __getitem__(self, i):
        return self._transforms[i]

    def __call__(self, x):
        for transform in self:
            x = transform(x)
        return x
