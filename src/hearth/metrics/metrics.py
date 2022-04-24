import sys
from dataclasses import dataclass

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

from torch import Tensor
from typing import Optional, Tuple
from hearth.metrics.base import Metric
from hearth.metrics.functional import pearson_corr
from hearth.metrics.mixins import (
    HardBinaryMixin,
    MaskingMixin,
    AccuracyMixin,
    ArgmaxMixin,
    RecallMixin,
    PrecisionMixin,
    F1Mixin,
    FBetaMixin,
    BinaryMixin,
    OneHotMixin,
    AverageMixin,
)


@dataclass
class BinaryAccuracy(HardBinaryMixin, MaskingMixin, AccuracyMixin):
    """binary accruracy over possibly unnormalized values (see ``from_logits``) with target masking\
     support.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.
        from_logits: if ``True`` inputs are expected to be unnormalized and a sigmoid
            function will be applied before comparison to targets. defaults to ``False``.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> from hearth.metrics import BinaryAccuracy
        >>>
        >>>
        >>> metric = BinaryAccuracy()
        >>> metric
        BinaryAccuracy(mask_target=-1, from_logits=False)

        by default ``from_logits`` is false so we expect inputs to be normalized.

        >>> targets = torch.randint(1, size=(100,)) # (batch, 1)
        >>> predictions = torch.normal(0, 1, size=(100, 1)) # (batch, 1)
        >>>
        >>> metric(predictions.sigmoid(), targets)
        tensor(0.4400)

        alternatively we could specify ``from_logits=True`` in which case inputs
        will be sigmoided for us before comparing with targets.

        >>> metric = BinaryAccuracy(from_logits=True)
        >>> metric(predictions, targets) # no sigmoid!
        tensor(0.4400)

        this metric also supports masked targets in the case of variable length sequence data
        which should be **batch first** ``(batch, time)`` or ``(batch, time, 1)``.

        >>> metric = BinaryAccuracy()
        >>> targets = torch.tensor([[1, 1, 1, -1], [1, 1, -1, -1], [1, 1, 1, 0]]) # (batch, time)
        >>> predictions = torch.ones(3, 4)
        >>> metric(predictions, targets)
        tensor(0.8889)
    """

    pass


@dataclass
class CategoricalAccuracy(MaskingMixin, ArgmaxMixin, AccuracyMixin):
    """categorical accuracy over possibly unnormalized scores given target indices.

    Built to have a similar interface to\
     `nn.CrossEntropyLoss <https://pytorch.org/docs/stable/data.html#torch.nn.CrossEntropyLoss>`_.

    Args:
        mask_target: mask this value if seen in the targets to this value. defaults to ``-1``

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> from hearth.metrics import CategoricalAccuracy
        >>>
        >>> metric = CategoricalAccuracy()
        >>> metric
        CategoricalAccuracy(mask_target=-1)


        >>> targets = torch.randint(4, size=(100,)) # (bath,)
        >>> predictions = torch.normal(0, 1, size=(100, 4)) # (batch, classes)
        >>> metric(predictions, targets)
        tensor(0.2200)


        mask targets using the ``mask_target``, particularly useful for mixed length
        sequence predicions....

        >>> targets = torch.tensor([[0, 5, 9, -1], [2, 3, -1, -1], [1, 6, 3, 4]]) # (batch, time)
        >>> predictions = torch.normal(0, 1, size=(3, 4,  9)) # (batch, time, classes)
        >>> metric(predictions, targets)
        tensor(0.1111)
    """

    pass


@dataclass
class BinaryRecall(HardBinaryMixin, MaskingMixin, RecallMixin):
    """"binary recall over possibly un-normalized values (see ``from_logits``) with target masking\
     support.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.
        from_logits: if ``True`` inputs are expected to be unnormalized and a sigmoid
            function will be applied before comparison to targets. defaults to ``False``.
    Example:
        >>> import torch
        >>> from hearth.metrics import BinaryRecall
        >>>
        >>> inputs = torch.tensor([0.7116, 0.6470, 0.5039, 0.9953, 0.8948, 0.4229, 0.8654, 0.8108])
        >>> targets = torch.tensor([0., 1., 1., 1., 0., 1., 0., 0.])
        >>>
        >>> recall = BinaryRecall()
        >>> recall(inputs, targets)
        tensor(0.7500)

        works fine with an extra dim:

        >>> recall(inputs.unsqueeze(-1), targets.unsqueeze(-1))
        tensor(0.7500)

        use the ``from_logits`` option if your inputs will not be sigmoid squashed:

        >>> recall = BinaryRecall(from_logits=True)
        >>> unsigmoided = torch.log(inputs/(1-inputs))
        >>> recall(unsigmoided, targets)
        tensor(0.7500)

        for cases like variable lenght time sequences use the mask target option:

        >>> recall = BinaryRecall(mask_target=-1)
        >>> inputs = inputs.reshape(2, 4) # (batch, sequence length)
        >>> # note we mask some timesteps with -1
        >>> targets = torch.tensor([[0., 1., 1., -1],
        ...                         [0., 1, -1, -1]])
        >>> recall(inputs, targets)
        tensor(0.6667)

    """

    pass


@dataclass
class BinaryPrecision(HardBinaryMixin, MaskingMixin, PrecisionMixin):
    """"binary precision over possibly un-normalized values (see ``from_logits``) with target \
    masking support.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.
        from_logits: if ``True`` inputs are expected to be unnormalized and a sigmoid
            function will be applied before comparison to targets. defaults to ``False``.
    Example:
        >>> import torch
        >>> from hearth.metrics import BinaryPrecision
        >>>
        >>> inputs = torch.tensor([0.7116, 0.6470, 0.5039, 0.9953, 0.8948, 0.4229, 0.8654, 0.8108])
        >>> targets = torch.tensor([0., 1., 1., 1., 0., 1., 0., 0.])
        >>>
        >>> precision = BinaryPrecision()
        >>> precision(inputs, targets)
        tensor(0.4286)

        works fine with an extra dim:

        >>> precision(inputs.unsqueeze(-1), targets.unsqueeze(-1))
        tensor(0.4286)

        use the ``from_logits`` option if your inputs will not be sigmoid squashed:

        >>> precision = BinaryPrecision(from_logits=True)
        >>> unsigmoided = torch.log(inputs/(1-inputs))
        >>> precision(unsigmoided, targets)
        tensor(0.4286)

        for cases like variable lenght time sequences use the mask target option:

        >>> precision = BinaryPrecision(mask_target=-1)
        >>> inputs = inputs.reshape(2, 4) # (batch, sequence length)
        >>> # note we mask some timesteps with -1
        >>> targets = torch.tensor([[0., 1., 1., -1],
        ...                         [0., 1, -1, -1]])
        >>> precision(inputs, targets)
        tensor(0.5000)
    """

    pass


@dataclass
class SoftBinaryRecall(BinaryMixin, MaskingMixin, RecallMixin):
    """a soft version of binary recall over possibly un-normalized values (see ``from_logits``)\
     with target masking support.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.
        from_logits: if ``True`` inputs are expected to be unnormalized and a sigmoid
            function will be applied before comparison to targets. defaults to ``False``.

    Example:
        >>> import torch
        >>> from hearth.metrics import SoftBinaryRecall
        >>>
        >>> inputs = torch.tensor([0.7116, 0.6470, 0.5039, 0.9953, 0.8948, 0.4229, 0.8654, 0.8108])
        >>> targets = torch.tensor([0., 1., 1., 1., 0., 1., 0., 0.])
        >>>
        >>> softrecall = SoftBinaryRecall()
        >>> softrecall(inputs, targets)
        tensor(0.6423)

        works fine with an extra dim:

        >>> softrecall(inputs.unsqueeze(-1), targets.unsqueeze(-1))
        tensor(0.6423)

        use the ``from_logits`` option if your inputs will not be sigmoid squashed:

        >>> softrecall = SoftBinaryRecall(from_logits=True)
        >>> unsigmoided = torch.log(inputs/(1-inputs))
        >>> softrecall(unsigmoided, targets)
        tensor(0.6423)

        for cases like variable lenght time sequences use the mask target option:

        >>> softrecall = SoftBinaryRecall(mask_target=-1)
        >>> inputs = inputs.reshape(2, 4) # (batch, sequence length)
        >>> # note we mask some timesteps with -1
        >>> targets = torch.tensor([[0., 1., 1., -1],
        ...                         [0., 1, -1, -1]])
        >>> softrecall(inputs, targets)
        tensor(0.5246)
    """

    pass


@dataclass
class SoftBinaryPrecision(BinaryMixin, MaskingMixin, PrecisionMixin):
    """a soft version of binary precision over possibly un-normalized values (see ``from_logits``)\
     with target masking support.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.
        from_logits: if ``True`` inputs are expected to be unnormalized and a sigmoid
            function will be applied before comparison to targets. defaults to ``False``.

    Example:
        >>> import torch
        >>> from hearth.metrics import SoftBinaryPrecision
        >>>
        >>> inputs = torch.tensor([0.7116, 0.6470, 0.5039, 0.9953, 0.8948, 0.4229, 0.8654, 0.8108])
        >>> targets = torch.tensor([0., 1., 1., 1., 0., 1., 0., 0.])
        >>>
        >>> softprecision = SoftBinaryPrecision()
        >>> softprecision(inputs, targets)
        tensor(0.4390)

        works fine with an extra dim:

        >>> softprecision(inputs.unsqueeze(-1), targets.unsqueeze(-1))
        tensor(0.4390)

        use the ``from_logits`` option if your inputs will not be sigmoid squashed:

        >>> softprecision = SoftBinaryPrecision(from_logits=True)
        >>> unsigmoided = torch.log(inputs/(1-inputs))
        >>> softprecision(unsigmoided, targets)
        tensor(0.4390)

        for cases like variable lenght time sequences use the mask target option:

        >>> softprecision = SoftBinaryPrecision(mask_target=-1)
        >>> inputs = inputs.reshape(2, 4) # (batch, sequence length)
        >>> # note we mask some timesteps with -1
        >>> targets = torch.tensor([[0., 1., 1., -1],
        ...                         [0., 1, -1, -1]])
        >>> softprecision(inputs, targets)
        tensor(0.4949)
    """

    pass


@dataclass
class BinaryF1(HardBinaryMixin, MaskingMixin, F1Mixin):
    """binary f1 score over possibly un-normalized values (see ``from_logits``) with target \
    masking support.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.
        from_logits: if ``True`` inputs are expected to be unnormalized and a sigmoid
            function will be applied before comparison to targets. defaults to ``False``.

    Example:
        >>> import torch
        >>> from hearth.metrics import BinaryF1
        >>>
        >>> inputs = torch.tensor([0.7116, 0.6470, 0.5039, 0.9953, 0.8948, 0.4229, 0.8654, 0.8108])
        >>> targets = torch.tensor([0., 1., 1., 1., 0., 1., 0., 0.])
        >>>
        >>> f1 = BinaryF1()
        >>> f1(inputs, targets)
        tensor(0.5455)

        works fine with an extra dim:

        >>> f1(inputs.unsqueeze(-1), targets.unsqueeze(-1))
        tensor(0.5455)

        use the ``from_logits`` option if your inputs will not be sigmoid squashed:

        >>> f1 = BinaryF1(from_logits=True)
        >>> unsigmoided = torch.log(inputs/(1-inputs))
        >>> f1(unsigmoided, targets)
        tensor(0.5455)

        for cases like variable lenght time sequences use the mask target option:

        >>> f1 = BinaryF1(mask_target=-1)
        >>> inputs = inputs.reshape(2, 4) # (batch, sequence length)
        >>> # note we mask some timesteps with -1
        >>> targets = torch.tensor([[0., 1., 1., -1],
        ...                         [0., 1, -1, -1]])
        >>> f1(inputs, targets)
        tensor(0.5714)
    """

    pass


@dataclass
class BinaryFBeta(HardBinaryMixin, MaskingMixin, FBetaMixin):
    """binary fbeta score over possibly un-normalized values (see ``from_logits``) with target \
    masking support.

    Args:
        beta: beta value for weighting precision and recall. ``beta < 1`` weights precision higher,
            while ``beta > 1`` gives more weight to recall. Defaults to 1 (making the default the
             same as f1 score).
        mask_target: mask targets equal to this value. defaults to ``-1``.
        from_logits: if ``True`` inputs are expected to be unnormalized and a sigmoid
            function will be applied before comparison to targets. defaults to ``False``.

    Example:
        >>> import torch
        >>> from hearth.metrics import BinaryFBeta
        >>>
        >>> inputs = torch.tensor([0.7116, 0.6470, 0.5039, 0.9953, 0.8948, 0.4229, 0.8654, 0.8108])
        >>> targets = torch.tensor([0., 1., 1., 1., 0., 1., 0., 0.])
        >>>
        >>> fbeta = BinaryFBeta(beta=.5)
        >>> fbeta(inputs, targets)
        tensor(0.4687)

        works fine with an extra dim:

        >>> fbeta(inputs.unsqueeze(-1), targets.unsqueeze(-1))
        tensor(0.4687)

        use the ``from_logits`` option if your inputs will not be sigmoid squashed:

        >>> fbeta = BinaryFBeta(beta=.5, from_logits=True)
        >>> unsigmoided = torch.log(inputs/(1-inputs))
        >>> fbeta(unsigmoided, targets)
        tensor(0.4687)

        for cases like variable lenght time sequences use the mask target option:

        >>> fbeta = BinaryFBeta(beta=.5, mask_target=-1)
        >>> inputs = inputs.reshape(2, 4) # (batch, sequence length)
        >>> # note we mask some timesteps with -1
        >>> targets = torch.tensor([[0., 1., 1., -1],
        ...                         [0., 1, -1, -1]])
        >>> fbeta(inputs, targets)
        tensor(0.5263)
    """

    pass


@dataclass
class CategoricalRecall(MaskingMixin, OneHotMixin, RecallMixin, AverageMixin):
    """categorical recall over possibly unnormalized scores given target indices.

    Built to have a somewhat similar interface to\
     `nn.CrossEntropyLoss <https://pytorch.org/docs/stable/data.html#torch.nn.CrossEntropyLoss>`_.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.

    Example:
        >>> import torch
        >>> from hearth.metrics import CategoricalRecall
        >>>
        >>> # here inputs is batch, feats and is not nessisarily normalized
        >>> inputs = torch.tensor([[-0.2956,  1.6050,  0.4113, -1.9041],
        ...                        [ 0.2095,  1.2959, -1.2466,  2.2302],
        ...                        [ 0.4702, -0.7506,  1.6751,  0.3370],
        ...                        [-0.4504,  0.5301, -1.1206, -0.5896],
        ...                        [ 0.7439,  0.4022,  0.5913,  0.1511],
        ...                        [-0.0523, -1.0082,  0.5536, -1.2748],
        ...                        [ 0.5151, -0.9396,  0.7223, -0.5500],
        ...                        [ 0.1083,  2.7311,  1.4429,  1.0640]])
        >>> targets = torch.tensor([3, 1, 1, 2, 0, 3, 0, 2]) # true class indices
        >>>
        >>> recall = CategoricalRecall()
        >>> recall(inputs, targets)
        tensor(0.1250)

        if you're predicting categories for variable length sequences
        you can mask your targets with ``mask_target`` (-1 by default)
        predictions at those timesteps will then be excluded from the metric.

        >>> sequence_inputs = inputs.reshape(2, 4, 4) # (batch, timesteps, classes)
        >>> # note the last two timesteps in targets are masked with -1
        >>> masked_targets = torch.tensor([[ 3,  1,  1, 2],
        ...                                [ 0, 3, -1, -1]]) # (batch, timesteps)
        >>> recall(sequence_inputs, masked_targets)
        tensor(0.2500)
    """

    pass


@dataclass
class CategoricalPrecision(MaskingMixin, OneHotMixin, PrecisionMixin, AverageMixin):
    """categorical precision over possibly unnormalized scores given target indices.

    Built to have a somewhat similar interface to\
     `nn.CrossEntropyLoss <https://pytorch.org/docs/stable/data.html#torch.nn.CrossEntropyLoss>`_.


    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.

    Example:
        >>> import torch
        >>> from hearth.metrics import CategoricalPrecision
        >>>
        >>> # here inputs is batch, feats and is not nessisarily normalized
        >>> inputs = torch.tensor([[-0.2956,  1.6050,  0.4113, -1.9041],
        ...                        [ 0.2095,  1.2959, -1.2466,  2.2302],
        ...                        [ 0.4702, -0.7506,  1.6751,  0.3370],
        ...                        [-0.4504,  0.5301, -1.1206, -0.5896],
        ...                        [ 0.7439,  0.4022,  0.5913,  0.1511],
        ...                        [-0.0523, -1.0082,  0.5536, -1.2748],
        ...                        [ 0.5151, -0.9396,  0.7223, -0.5500],
        ...                        [ 0.1083,  2.7311,  1.4429,  1.0640]])
        >>> targets = torch.tensor([3, 1, 1, 2, 0, 3, 0, 2]) # true class indices
        >>>
        >>> precision = CategoricalPrecision()
        >>> precision(inputs, targets)
        tensor(0.2500)

        if you're predicting categories for variable length sequences
        you can mask your targets with ``mask_target`` (-1 by default)
        predictions at those timesteps will then be excluded from the metric.

        >>> sequence_inputs = inputs.reshape(2, 4, 4) # (batch, timesteps, classes)
        >>> # note the last two timesteps in targets are masked with -1
        >>> masked_targets = torch.tensor([[ 3,  1,  1, 2],
        ...                                [ 0,  3, -1, -1]]) # (batch, timesteps)
        >>> precision(sequence_inputs, masked_targets)
        tensor(0.2500)
    """

    pass


@dataclass
class CategoricalFBeta(MaskingMixin, OneHotMixin, FBetaMixin, AverageMixin):
    """categorical fbeta score over possibly unnormalized scores given target indices.

    Built to have a somewhat similar interface to\
     `nn.CrossEntropyLoss <https://pytorch.org/docs/stable/data.html#torch.nn.CrossEntropyLoss>`_.

    Args:
        beta: beta value for weighting precision and recall.
            ``beta < 1`` weights precision higher, while ``beta > 1`` gives
            more weight to recall. Defaults to 1 (making the default the same as f1 score).
        mask_target: mask targets equal to this value. defaults to ``-1``.

    Example:
        >>> import torch
        >>> from hearth.metrics import CategoricalFBeta
        >>>
        >>> # here inputs is batch, feats and is not nessisarily normalized
        >>> inputs = torch.tensor([[-0.2956,  1.6050,  0.4113, -1.9041],
        ...                        [ 0.2095,  1.2959, -1.2466,  2.2302],
        ...                        [ 0.4702, -0.7506,  1.6751,  0.3370],
        ...                        [-0.4504,  0.5301, -1.1206, -0.5896],
        ...                        [ 0.7439,  0.4022,  0.5913,  0.1511],
        ...                        [-0.0523, -1.0082,  0.5536, -1.2748],
        ...                        [ 0.5151, -0.9396,  0.7223, -0.5500],
        ...                        [ 0.1083,  2.7311,  1.4429,  1.0640]])
        >>> targets = torch.tensor([3, 1, 1, 2, 0, 3, 0, 2]) # true class indices
        >>>
        >>> fbeta = CategoricalFBeta(beta=.5)
        >>> fbeta(inputs, targets)
        tensor(0.2083)

        if you're predicting categories for variable length sequences
        you can mask your targets with ``mask_target`` (-1 by default)
        predictions at those timesteps will then be excluded from the metric.

        >>> sequence_inputs = inputs.reshape(2, 4, 4) # (batch, timesteps, classes)
        >>> # note the last two timesteps in targets are masked with -1
        >>> masked_targets = torch.tensor([[ 3,  1,  1, 2],
        ...                                [ 0,  3, -1, -1]]) # (batch, timesteps)
        >>> fbeta(sequence_inputs, masked_targets)
        tensor(0.2500)
    """

    pass


@dataclass
class CategoricalF1(MaskingMixin, OneHotMixin, F1Mixin, AverageMixin):
    """categorical f1 score over possibly unnormalized scores given target indices.

    Built to have a somewhat similar interface to\
     `nn.CrossEntropyLoss <https://pytorch.org/docs/stable/data.html#torch.nn.CrossEntropyLoss>`_.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.

    Example:
        >>> import torch
        >>> from hearth.metrics import CategoricalF1
        >>>
        >>> # here inputs is batch, feats and is not nessisarily normalized
        >>> inputs = torch.tensor([[-0.2956,  1.6050,  0.4113, -1.9041],
        ...                        [ 0.2095,  1.2959, -1.2466,  2.2302],
        ...                        [ 0.4702, -0.7506,  1.6751,  0.3370],
        ...                        [-0.4504,  0.5301, -1.1206, -0.5896],
        ...                        [ 0.7439,  0.4022,  0.5913,  0.1511],
        ...                        [-0.0523, -1.0082,  0.5536, -1.2748],
        ...                        [ 0.5151, -0.9396,  0.7223, -0.5500],
        ...                        [ 0.1083,  2.7311,  1.4429,  1.0640]])
        >>> targets = torch.tensor([3, 1, 1, 2, 0, 3, 0, 2]) # true class indices
        >>>
        >>> f1 = CategoricalF1()
        >>> f1(inputs, targets)
        tensor(0.1667)

        if you're predicting categories for variable length sequences
        you can mask your targets with ``mask_target`` (-1 by default)
        predictions at those timesteps will then be excluded from the metric.

        >>> sequence_inputs = inputs.reshape(2, 4, 4) # (batch, timesteps, classes)
        >>> # note the last two timesteps in targets are masked with -1
        >>> masked_targets = torch.tensor([[ 3,  1,  1, 2],
        ...                                [ 0, 3, -1, -1]]) # (batch, timesteps)
        >>> f1(sequence_inputs, masked_targets)
        tensor(0.2500)
    """

    pass


@dataclass
class PearsonCorrCoef(Metric):
    """pearson correlation coefficient with optional input transform.

    Args:
        transform_inputs: one of ['sigmoid', 'tanh', None]. If populated apply this transform
            to inputs before computing metric. Defaults to None ( No Transform).

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> from hearth.metrics import PearsonCorrCoef
        >>>
        >>>
        >>> metric = PearsonCorrCoef()

        by default ``transform_inputs=None``.

        >>> targets = torch.normal(0, 1, size=(10, 1)) # (batch, 1)
        >>> predictions = torch.normal(0, 1, size=(10, 1)) # (batch, 1)
        >>> metric(predictions, targets)
        tensor(0.3518)

        also works if extra dim is removed:

        >>> metric(predictions.squeeze(), targets.squeeze())
        tensor(0.3518)

        we can specify sigmoid transform...

        >>> metric = PearsonCorrCoef(transform_inputs='sigmoid')
        >>> metric(torch.normal(0, 1, size=(5,)), torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0]))
        tensor(0.3267)

        or tanh transform....

        >>> metric = PearsonCorrCoef(transform_inputs='sigmoid')
        >>> metric(torch.normal(0, 1, size=(5,)), torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0]))
        tensor(0.2911)

    """

    transform_inputs: Optional[Literal['sigmoid', 'tanh']] = None

    def _prepare(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        targets = targets.squeeze(-1)
        inputs = inputs.reshape_as(targets)
        if self.transform_inputs == 'sigmoid':
            inputs = inputs.sigmoid()
        elif self.transform_inputs == 'tanh':
            inputs = inputs.tanh()
        return inputs, targets

    def forward(self, inputs: Tensor, targets: Tensor, **kwargs) -> Tensor:  # type : ignore
        return pearson_corr(inputs, targets)
