import torch
from torch import nn
from hearth.modules.base import BaseModule


class Residual(BaseModule):
    """wraps a block in a residual connection :math:`y = block(x) + x`.

    Args:
        block: the module to wrap.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.modules import Residual
        >>> _ = torch.manual_seed(0)
        >>>
        >>> res = Residual(nn.Linear(4, 4))
        >>>
        >>> x = torch.rand(2, 4) # (batch, feats)
        >>> res(x)
        tensor([[ 0.6371,  1.5493,  0.0031, -0.0379],
                [ 0.3584,  0.8512,  0.5208, -0.7607]], grad_fn=<AddBackward0>)
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward padd for ``Residual`` wrapper."""
        y = self.block(x)
        return x + y


class ReZero(Residual):
    """Implements ReZero residual connection around a block with dropout (as in transformer\
     implementation).

    **Reference**:
        `Bachlechner et al: ReZero is All You Need: Fast Convergence at Large Depth
        <https://arxiv.org/abs/2003.04887>`_

    Example:
        >>> import torch
        >>> from hearth.modules import ReZero
        >>>
        >>> transformation = nn.Sequential(nn.Linear(10, 10),
        ...                                nn.ReLU(),
        ...                                nn.Dropout(.1),
        ...                                nn.Linear(10, 10)
        ...                                )
        >>> re_zero = ReZero(transformation, dropout=.1)
        >>>
        >>> x = torch.normal(0, 1, size=(5, 10))
        >>> y = re_zero(x)
        >>> y.shape
        torch.Size([5, 10])

        since ``re_zero``'s weight parameter has not actually been trained ``y`` its going to be
        equal to ``x`` and nothing from the transformation will be added to the input...
        as training goes on this should change.

        >>> (y == x).all()
        tensor(True)
    """

    def __init__(self, block: nn.Module, dropout: float = 0.0):
        super().__init__(block=block)
        self.dropout = nn.Dropout(dropout)
        self.res_weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward for ReZero."""
        y = self.block(x)
        y = y * self.res_weight
        return x + self.dropout(y)


class TimeMasked(BaseModule):
    """Call the wrapped layer such that only valid timesteps are passed to the underlying layer.

    Inputs are expected to be of shape (B, T, ...) and mask is expected to be a boolean mask of
    of shape (B, T) where **valid** timesteps are True and invalid (padding) timesteps are false
    underlying layer is expected to accept a single input of the masked shape.

    Example:
        >>> from torch import nn
        >>> from hearth.modules import TimeMasked
        >>>
        >>> layer = TimeMasked(nn.Sequential(nn.Linear(8, 12), nn.LayerNorm(12)))
        >>> mask = torch.tensor([[ True,  True,  True, False, False],
        ...                      [ True,  True,  True,  True,  True]])
        >>> inp = torch.rand(2, 5, 8)
        >>> out = layer(inp, mask)
        >>> out.shape
        torch.Size([2, 5, 12])

        >>> (out == 0.0).all(-1)
        tensor([[False, False, False,  True,  True],
               [False, False, False, False, False]])

    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def blocks(self):
        if hasattr(self.layer, 'blocks'):
            yield from self.layer.blocks()
        yield self.layer

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """call the underlying layer using only timesteps from `mask` padding \
        invalid timesteps with 0.

        Args:
            x: a tensor of shape (B, T, ...)
            mask: boolean mask of of shape (B, T) where **valid** timesteps are True and
                invalid (padding) timesteps are false.
        """
        batch, timesteps = mask.shape
        masked_out = self.layer(x[mask])
        out = masked_out.new_zeros(batch, timesteps, *masked_out.shape[1:])
        out[mask] += masked_out
        return out
