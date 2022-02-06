from typing import Optional, Union, Dict, Mapping, Callable
import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss


from hearth.containers import TensorDict, NumberDict
from hearth._multihead import _MultiHeadFunc


def _identity(x):
    return x


class MultiHeadLoss(nn.Module, _MultiHeadFunc):
    """wrapper for losses for models with multiple output heads.

    Args:
        weights: a mapping of key to scalar weight, will be multiplied with losses before summing
            to create the aggregate value at `aggregate_key`, need not sum to 1. Defaults to None
            (all losses weighted evenly).
        aggregate_key: the key to use for the aggregate loss. Defaults to 'weighted_sum'.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.losses import MultiHeadLoss
        >>>
        >>> _ =torch.manual_seed(0)
        >>>
        >>> loss = MultiHeadLoss(a=nn.BCEWithLogitsLoss(),
        ...                      b=nn.CrossEntropyLoss())
        >>> loss
        MultiHeadLoss(a=BCEWithLogitsLoss(),
                      b=CrossEntropyLoss(),
                      weights=NumberDict({'a': 1.0, 'b': 1.0}),
                      aggregate_key=weighted_sum)

        multihead loss expects inputs and targets to be dicts with containing
        the it's keys (in this case 'a' and 'b'):

        >>> batch_size = 10
        >>> inputs = {'a': torch.rand(batch_size, 1),
        ...           'b': torch.normal(batch_size, 1, size=(batch_size, 4))}
        >>> targets = {'a': torch.rand(batch_size, 1).round(),
        ...            'b': torch.randint(4, size=(batch_size,))}
        >>>
        >>> loss(inputs, targets)
        TensorDict({'a': tensor(0.5791), 'b': tensor(1.2425), 'weighted_sum': tensor(1.8216)})

        ``weighted_sum`` is the default aggregate key. this is the bit you should call backward on.
        you can change the default aggregate key if you like by specifying it at init.

        >>> loss = MultiHeadLoss(a=nn.BCEWithLogitsLoss(),
        ...                      b=nn.CrossEntropyLoss(),
        ...                      aggregate_key='sally')
        >>> loss(inputs, targets)
        TensorDict({'a': tensor(0.5791), 'b': tensor(1.2425), 'sally': tensor(1.8216)})

        you can aslo specify ``weights`` at init to weight contribution losses differently:

        >>> loss = MultiHeadLoss(a=nn.BCEWithLogitsLoss(),
        ...                      b=nn.CrossEntropyLoss(),
        ...                      weights={'a': .2, 'b':.8})
        >>> loss(inputs, targets)
        TensorDict({'a': tensor(0.5791), 'b': tensor(1.2425), 'weighted_sum': tensor(1.1098)})
    """

    def __init__(
        self,
        *,
        weights: Optional[Union[NumberDict, Dict[str, int]]] = None,
        aggregate_key: str = 'weighted_sum',
        **kwargs,
    ):
        nn.Module.__init__(self)
        self._fns = nn.ModuleDict(kwargs)
        self.aggregate_key = aggregate_key
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: Optional[Union[NumberDict, Dict[str, int]]] = None):
        if weights is not None:
            if weights is not NumberDict:
                weights = NumberDict(weights)
            if set(weights.keys()) != set(self.keys()):
                raise ValueError('weight keys must match keys for loss functions!')
        else:
            weights = NumberDict({k: 1.0 for k in self.keys()})
        self._weights = weights

    def _aggregate(self, out):
        out[self.aggregate_key] = (out * self.weights).sum()
        return out

    def forward(
        self, inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], **kwargs
    ) -> TensorDict:
        out = TensorDict()
        for k, func in self.items():
            out[k] = func(inputs[k], targets[k], **kwargs)
        return self._aggregate(out)

    def _argrepr(self):
        return (
            super()._argrepr() + f', weights={self.weights!r}, aggregate_key={self.aggregate_key}'
        )

    def __repr__(self):
        return f'{self.__class__.__name__}({self._argrepr()})'


class _BaseLoss(nn.Module):
    """base class for losses."""

    _reductions: Dict[str, Callable] = {'mean': torch.mean, 'sum': torch.sum, 'none': _identity}

    def __init__(self, *args, reduction: str, **kwargs):
        super().__init__()
        if reduction not in self._reductions:
            raise ValueError(
                f'reduction {reduction!r} is not supported for {self.__class__.__name__},'
                f' please choose one of {list(self._reductions)!r}'
            )
        self.reduction = reduction
        self._reduce_fn = self._reductions[reduction]

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self._reduce_fn(x)

    def extra_repr(self) -> str:
        return f'reduction={self.reduction}'


class _MaskedLoss(_BaseLoss):
    """base class for losses that support masking by target value."""

    def __init__(self, *args, mask_target_value: int = -1, reduction: str, **kwargs):
        super().__init__(*args, reduction=reduction, **kwargs)
        self.mask_target_value = mask_target_value

    def _masked_reduce(self, x, mask: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'none':
            return x * (mask * 1.0)
        return self._reduce_fn(x[mask])

    def _get_mask(self, targets: torch.Tensor) -> torch.Tensor:
        return targets != self.mask_target_value

    def extra_repr(self) -> str:
        return f'mask_target_value={self.mask_target_value}, reduction={self.reduction!r}'


class MulticlassFocalLoss(_MaskedLoss):
    """multiclass focal loss.

    Focal loss is a proposed solution for handling class imbalambce in segmentation.

    Reference:
       `Li et al. : Focal Loss for Dense Object Detection <https://arxiv.org/abs/1708.02002>`_

    Args:
        alpha: class weighting factor, may be a scalar, or a tensor with one weight per class
            if None alpha=1.0. Defaults to None.
        gamma: focusing factor. Defaults to 2.0.
        mask_target_value: this index will be masked when seen in the targets upon
            computing the loss. Defaults to -1.
        reduction: string . Defaults to 'mean'.

    Shape:
        - inputs: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
        - targets: :math:`(N, *, C)`, where `C = number of classes`.
        - output: scalar unless :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same shape as\
            targets.

    Example:
        >>> import torch
        >>> from hearth.losses import MulticlassFocalLoss
        >>>
        >>> inp = torch.tensor([[ 1.3053, -0.6421, -1.2027,  1.0494, -1.9540],
        ...                     [ 0.6801,  0.2266,  0.8120,  0.9490, -0.8120],
        ...                     [-0.8212,  0.8024,  0.8370, -0.3272,  0.6125],
        ...                     [-0.1975, -1.0706,  2.6819, -0.4297,  0.1980],
        ...                     [ 0.8256,  1.7839, -1.5876,  1.7705, -1.7051],
        ...                     [ 0.1288,  1.0981,  0.0570, -1.1684,  0.4567],
        ...                     [ 0.5658,  1.3948, -1.1457, -0.5921, -0.8026],
        ...                     [-0.3989,  0.6574,  0.3411, -1.9814,  0.2935]])
        >>>
        >>> targets = torch.tensor([0, 1, 4, 2, 3, 0, 2, 2])
        >>> loss = MulticlassFocalLoss()
        >>> loss
        MulticlassFocalLoss(alpha=1.0, gamma=2.0, mask_target_value=-1, reduction='mean')

        >>> loss(inp, targets)
        tensor(0.9478)

        pass class weights for the alpha value:

        >>> weights = torch.tensor([1.0, 2.0, 0.3, 2.1, .5])
        >>> loss = MulticlassFocalLoss(alpha=weights)
        >>> loss(inp, targets)
        tensor(0.8010)

        supports masking by value for instance if we want to predict classes over time and the last\
         two timesteps for the last batch are missing we can mask them by setting to \
         :attr:`mask_target_value` to -1 and padding targets with -1:

        >>> masked_targets =  torch.tensor([[0, 1, 4, 2], [3, 0, -1, -1]]) # (batch, time)
        >>> inp = inp.reshape((2, 4, -1)) # (batch, time, classes)
        >>> loss(inp, masked_targets)
        tensor(0.8885)

        when :attr:`reduction` is ``'none'`` we will return one loss value per step as you can see\
         the masked targets are 0.0:

        >>> loss = MulticlassFocalLoss(alpha=weights, reduction='none')
        >>> loss(inp, masked_targets)
        tensor([[1.8430e-01, 2.7830e+00, 4.0199e-01, 1.6719e-03],
                [6.7119e-01, 1.2889e+00, 0.0, 0.0]])
    """

    def __init__(
        self,
        alpha: Optional[Union[torch.Tensor, float]] = None,
        gamma: float = 2.0,
        mask_target_value: int = -1,
        reduction: str = 'mean',
    ):
        super().__init__(mask_target_value=mask_target_value, reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: Union[torch.Tensor, float, None]):
        if alpha is None:
            alpha = 1.0
        if isinstance(alpha, float):
            self._alpha = alpha
            self._scalar_alpha = True
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer('_alpha', alpha)
            self._scalar_alpha = False

    def _get_alphas(self, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self._scalar_alpha:
            return self.alpha

        alpha_t = self.alpha[targets]
        return alpha_t * (mask * 1.0)

    def _get_ce(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(inputs.shape) > 2:
            # this torch expects (batch, classes, ...)
            # where we expect (batch, ..., classes)
            # so we transpose before throwing in
            inputs = inputs.transpose(1, -1)

        return nn.functional.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.mask_target_value
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self._get_ce(inputs, targets)
        mask = self._get_mask(targets)
        p_t = torch.exp(-ce)
        alpha_t = self._get_alphas(targets, mask)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce
        return self._masked_reduce(focal_loss, mask=mask)

    def extra_repr(self) -> str:
        parent_args = super().extra_repr()
        return f'alpha={self.alpha!r}, gamma={self.gamma}, {parent_args}'


class BinaryFocalLoss(_MaskedLoss):
    """Binary focal loss.

    Focal loss is a proposed solution for handling class imbalambce in segmentation.

    Reference:
       `Li et al. : Focal Loss for Dense Object Detection <https://arxiv.org/abs/1708.02002>`_

    Args:
        alpha: class weighting factor. Defaults to 0.25 (defaults taken from paper).
        gamma: focusing factor. Defaults to 2.0 (default taken from paper).
        mask_target_value: this index will be masked when seen in the targets upon
            computing the loss. Defaults to -1.
        reduction: string name of reduction. Defaults to 'mean'.

    Shape:
        - inputs: :math:`(N, *)` where :math:`*` means, any number of additional dimensions\
            if inputs have an extra single dimension thats ok... they will be reshaped as targets.
        - targets: :math:`(N, *)`.
        - output: scalar unless :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same shape as\
            targets.

    Example:
        >>> import torch
        >>> from hearth.losses import BinaryFocalLoss
        >>>
        >>> targets = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.float32)
        >>> inputs = torch.tensor([-1.1645,  -0.2928, -0.5685, -0.8038, -0.0211,  2.0062])
        >>> loss = BinaryFocalLoss()
        >>> loss
        BinaryFocalLoss(alpha=0.25, gamma=2.0, mask_target_value=-1, reduction='mean')

        >>> loss(inputs, targets)
        tensor(0.2399)

        if your inputs have an extra dim thats ok:

        >>> loss(inputs.unsqueeze(-1), targets)
        tensor(0.2399)

        supports masking by value for instance if we want to predict a binary value over time and \
        the last timestep for the last example is missing, we can mask by setting \
        :attr:`mask_target_value` to -1 and padding targets with -1:

        >>> masked_targets =  torch.tensor([[0, 1, 0], [0, 1, -1]], dtype=torch.float32)
        >>> inputs = inputs.reshape((2, 3, -1)) # (batch, time, 1)
        >>> loss(inputs, masked_targets)
        tensor(0.0393)

        when :attr:`reduction` is ``'none'`` we will return one loss value per step as you can see\
         the masked targets are 0.0:

        >>> loss = BinaryFocalLoss(reduction='none')
        >>> loss(inputs, masked_targets)
        tensor([[0.0115, 0.0697, 0.0440],
                [0.0265, 0.0449, 0.0000]])
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        mask_target_value: int = -1,
        reduction: str = 'mean',
    ):
        super().__init__(mask_target_value=mask_target_value, reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.reshape_as(targets)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        a_t = (1 - self.alpha) + targets * (2 * self.alpha - 1)
        p_t = torch.exp(-bce)
        focal = a_t * (1 - p_t) ** self.gamma * bce
        return self._masked_reduce(focal, mask=self._get_mask(targets))

    def extra_repr(self) -> str:
        parent_args = super().extra_repr()
        return f'alpha={self.alpha!r}, gamma={self.gamma}, {parent_args}'


class MaskedMSELoss(_Loss):
    """MSELoss with support for masked targets.

    Args:
        mask_target_value: ignore targets with this value. Defaults to -inf.
        reduction: Defaults to 'mean'.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> from hearth.losses import MaskedMSELoss
        >>>
        >>> ninf = -float('inf')
        >>> loss = MaskedMSELoss()
        >>> inputs = torch.rand(3, 5) # (batch, timesteps)
        >>> targets = torch.tensor([[ 1.1721,  0.3909, -5.2731,    ninf,   ninf],
        ...                         [ 2.4388,  2.5159, -1.0815, -1.9472, -0.5450],
        ...                         [ 4.0665, -2.5141,    ninf,    ninf,   ninf]])
        >>> loss(inputs, targets)
        tensor(7.0101)
    """

    def __init__(self, mask_target_value=-float('inf'), reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.mask_target_value = mask_target_value

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mask = target != self.mask_target_value
        return nn.functional.mse_loss(input[mask], target[mask], reduction=self.reduction)


class _AngularPenaltySoftMarginLoss(_BaseLoss):
    """base class for angular penalty soft margin losses."""

    def __init__(
        self,
        embedding_features: int,
        n_classes: int,
        scale: float,
        margin: float,
        eps: float = 1e-7,
        reduction='mean',
    ):
        super().__init__(reduction=reduction)
        self.embedding_features = embedding_features
        self.n_classes = n_classes
        self.scale = scale
        self.margin = margin
        self.eps = eps
        self.weight = nn.Parameter(torch.FloatTensor(self.n_classes, self.embedding_features))
        nn.init.xavier_uniform_(self.weight)

    def _get_cos(self, inputs: Tensor) -> Tensor:
        return nn.functional.linear(
            nn.functional.normalize(inputs, p=2, dim=-1),
            nn.functional.normalize(self.weight, p=2, dim=-1),
        )

    def _compute_unscaled_numerator(self, pos: Tensor):
        return NotImplemented

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        cos = self._get_cos(inputs)
        n_classes = cos.shape[-1]
        one_hot = torch.nn.functional.one_hot(targets, n_classes)
        numerator = self.scale * self._compute_unscaled_numerator((cos * one_hot).sum(-1))
        neg = cos.masked_fill(one_hot == 1, -float('inf'))
        denominator = numerator.exp() + (self.scale * neg).exp().sum(-1)
        err = -(numerator - torch.log(denominator))
        return self._reduce(err)

    def extra_repr(self) -> str:
        out = [
            f'embedding_features={self.embedding_features}',
            f'n_classes={self.n_classes}',
            f'scale={self.scale}',
            f'margin={self.margin}',
            f'eps={self.eps}',
            f'reduction={self.reduction!r}',
        ]

        return ', '.join(out)


class AdditiveAngularMarginLoss(_AngularPenaltySoftMarginLoss):
    """AKA ArcFaceLoss.

    Reference:
       `Deng et al. : ArcFace: Additive Angular Margin Loss for Deep Face Recognition\
            <https://arxiv.org/abs/1801.07698>`_

    Note:
        - this loss has weights and requires an optimizer.
        - :attr:`margin` is already expressed in radians.

    Args:
        embedding_features: number of embedding features inputs are expected to be
            (batch, embedding_features).
        n_classes: number of classes in projection.
        scale: Defaults to 64.0.
        margin: margin in radians. Defaults to 0.5 (28.6 degrees).
        eps: epsilon for clamping . Defaults to 1e-7.
        reduction: batch reduction for this loss should be one of 'mean', 'sum', 'none'.
            Defaults to 'mean'.

    Example:
        >>> import torch
        >>> from hearth.losses import AdditiveAngularMarginLoss
        >>> _ = torch.manual_seed(666)
        >>>
        >>>
        >>> # this would be embeddings coming out of your model...
        >>> emb = torch.normal(0, 1, size=(5, 128))
        >>> targets = torch.randint(0, 10, size=(5,))
        >>> loss = AdditiveAngularMarginLoss(128, 10)
        >>> loss(emb, targets)
        tensor(41.8191, grad_fn=<MeanBackward0>)
    """

    def __init__(
        self,
        embedding_features: int,
        n_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
        **kwargs,
    ):
        super().__init__(embedding_features, n_classes, scale=scale, margin=margin, **kwargs)

    def _compute_unscaled_numerator(self, pos: Tensor) -> Tensor:
        clipped = torch.clamp(pos, -1.0 + self.eps, 1 - self.eps)
        return torch.cos(clipped.acos() + self.margin)


class LargeMarginCosineLoss(_AngularPenaltySoftMarginLoss):
    """AKA CosFaceLoss.

    Reference:
       `Wang et al. : CosFace: Large Margin Cosine Loss for Deep Face Recognition\
            <https://arxiv.org/abs/1801.09414>`_

    Note:
        - this loss has weights and requires an optimizer.

    Args:
        embedding_features: number of embedding features inputs are expected to be
            (batch, embedding_features).
        n_classes: number of classes in projection.
        scale: Defaults to 30.0.
        margin: Defaults to 0.4.
        eps: epsilon for clamping . Defaults to 1e-7.
        reduction: batch reduction for this loss should be one of 'mean', 'sum', 'none'.
            Defaults to 'mean'.

    Example:
        >>> import torch
        >>> from hearth.losses import LargeMarginCosineLoss
        >>> _ = torch.manual_seed(666)
        >>>
        >>>
        >>> # this would be embeddings coming out of your model...
        >>> emb = torch.normal(0, 1, size=(5, 128))
        >>> targets = torch.randint(0, 10, size=(5,))
        >>> loss = LargeMarginCosineLoss(128, 10)
        >>> loss(emb, targets)
        tensor(17.5245, grad_fn=<MeanBackward0>)
    """

    def __init__(
        self,
        embedding_features: int,
        n_classes: int,
        scale: float = 30.0,
        margin: float = 0.4,
        **kwargs,
    ):
        super().__init__(embedding_features, n_classes, scale=scale, margin=margin, **kwargs)

    def _compute_unscaled_numerator(self, pos: Tensor) -> Tensor:
        return pos - self.margin


class SphereEmbeddingLoss(_AngularPenaltySoftMarginLoss):
    """AKA SphereFaceLoss.

    Reference:
       `Liu et al. : SphereFace: Deep Hypersphere Embedding for Face Recognition\
            <https://arxiv.org/abs/1704.08063>`_

    Note:
        - this loss has weights and requires an optimizer.

    Args:
        embedding_features: number of embedding features inputs are expected to be
            (batch, embedding_features).
        n_classes: number of classes in projection.
        scale: Defaults to 64.0.
        margin: Defaults to 1.35.
        eps: epsilon for clamping . Defaults to 1e-7.
        reduction: batch reduction for this loss should be one of 'mean', 'sum', 'none'.
            Defaults to 'mean'.

    Example:
        >>> import torch
        >>> from hearth.losses import SphereEmbeddingLoss
        >>> _ = torch.manual_seed(666)
        >>>
        >>>
        >>> # this would be embeddings coming out of your model...
        >>> emb = torch.normal(0, 1, size=(5, 128))
        >>> targets = torch.randint(0, 10, size=(5,))
        >>> loss = SphereEmbeddingLoss(128, 10)
        >>> loss(emb, targets)
        tensor(44.7385, grad_fn=<MeanBackward0>)
    """

    def __init__(
        self,
        embedding_features: int,
        n_classes: int,
        scale: float = 64.0,
        margin: float = 1.35,
        **kwargs,
    ):
        super().__init__(embedding_features, n_classes, scale=scale, margin=margin, **kwargs)

    def _compute_unscaled_numerator(self, pos: Tensor) -> Tensor:
        clipped = torch.clamp(pos, -1.0 + self.eps, 1 - self.eps)
        return torch.cos(self.margin * clipped.acos())
