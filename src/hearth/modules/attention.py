from typing import Optional, Tuple
import math

import torch
from torch import nn
from hearth.modules.base import BaseModule
from hearth.modules.wrappers import TimeMasked


class SelfAttention(BaseModule):
    """Scaled self attention layer for timeseries.

    Args:
        in_features: number of input features.
        out_features: number of output features, if None defaults to in_features. Defaults to None.
        n_heads: number of attention heads, must evenly divide out_features. Defaults to 1.
        bias: include bias in all layers. Defaults to True.
        dropout: attention dropout rate. Defaults to 0.1.
        drop_head: If true drop attention from entire attention heads as dropout strategy otherwise
            drop timesteps from different heads. Defaults to False.

    Example:
        >>> from torch import nn
        >>> from hearth.modules import SelfAttention
        >>>
        >>> layer = SelfAttention(16, 32, n_heads=4)
        >>> mask = torch.tensor([[ True,  True,  True, False, False],
        ...                      [ True,  True,  True,  True,  True]])
        >>> inp = torch.rand(2, 5, 16)
        >>> out = layer(inp, mask)
        >>> out.shape
        torch.Size([2, 5, 32])

        >>> (out == 0.0).all(-1)
        tensor([[False, False, False,  True,  True],
               [False, False, False, False, False]])

    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        n_heads: int = 1,
        bias: bool = True,
        dropout=0.1,
        drop_head: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features or in_features
        if self.out_features % n_heads != 0:
            raise ValueError('out_features must be divisible by n_heads')
        self.n_heads = n_heads
        self.dim_per_head = int(self.out_features // self.n_heads)
        # stack the whole goddamn thing together...
        self.qkv = TimeMasked(nn.Linear(self.in_features, 3 * self.out_features, bias=bias))
        self.scale = math.sqrt(self.dim_per_head)
        self.dropout = nn.Dropout2d(dropout) if drop_head else nn.Dropout(dropout)
        self.output = TimeMasked(nn.Linear(self.out_features, self.out_features, bias=bias))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1, self.n_heads, self.dim_per_head).transpose(1, 2)

    def _unshape(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.out_features)

    def _get_qkv(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.qkv(x, mask).chunk(3, -1)
        return self._shape(q), self._shape(k), self._shape(v)

    def _compute_attn(self, q, k, mask):
        b = mask.shape[0]
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        padding_mask = (~mask).view((b, 1, 1, -1)).expand_as(scores)
        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = weights.masked_fill(
            padding_mask.transpose(-1, 2), 0.0
        )  # (bs, n_heads, q_length, k_length)
        return weights

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self._get_qkv(x, mask)
        q = q / self.scale
        attn = self.dropout(self._compute_attn(q, k, mask))
        context = torch.matmul(attn, v)  # (bs, n_heads, q_length, dim_per_head)
        return self.output(self._unshape(context), mask)
