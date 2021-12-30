from typing import Union, List

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from hearth.modules import BaseModule


class LayerNormSimple(BaseModule):
    """Layer norm without learnable bias and gain.

    This basically just wraps the standard torch LayerNorm so it has no elementwise
    affine by default.

    Args:
        normalized_shape: input shape from an expected input. If a single integer is used, it is
             treated as a singleton list, and this module will normalize over the last dimension
             which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5

    **Reference**:
        `Xu et al: Understanding and Improving Layer Normalization
        <https://arxiv.org/abs/1911.07013>`_

    Example:
        >>> import torch
        >>> from hearth.modules import LayerNormSimple
        >>>
        >>> layer = LayerNormSimple(5) # 5 feats
        >>> layer
        LayerNormSimple(5, eps=1e-05)

        >>> x = torch.tensor([[-4.2721, -2.8159, -1.2351,  0.2388,  4.5915],
        ...                  [-0.9092, -3.9666, -1.4216, -4.7373,  2.0403],
        ...                  [-1.2210,  4.4796, -1.2772, -2.8781,  4.1868]])
        >>> layer(x)
        tensor([[-1.1730, -0.6950, -0.1761,  0.3077,  1.7365],
                [ 0.3694, -0.9000,  0.1566, -1.2200,  1.5940],
                [-0.6139,  1.2486, -0.6323, -1.1554,  1.1530]])

    """

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    def __repr__(self) -> str:
        shape_or_feats = self.norm.normalized_shape
        if len(shape_or_feats) == 1:  # type: ignore
            shape_or_feats = shape_or_feats[0]  # type: ignore
        return f'{self.__class__.__name__}({shape_or_feats}, eps={self.norm.eps})'


class MaskedLayerNorm(nn.LayerNorm):

    """layer norm that supports masked input.

    Only calcluates stats and normalizes non-padding timesteps.

    Inputs are expected to be of shape (B, T, ...) and mask is expected to be a boolean mask of
    of shape (B, T) where **valid** timesteps are True and invalid (padding) timesteps are false.

    Example:
        >>> from torch import nn
        >>> from hearth.modules import MaskedLayerNorm
        >>>
        >>> layer = MaskedLayerNorm(16)
        >>> mask = torch.tensor([[ True,  True,  True, False, False],
        ...                      [ True,  True,  True,  True,  True]])
        >>> inp = torch.rand(2, 5, 16)
        >>> out = layer(inp, mask)
        >>> out.shape
        torch.Size([2, 5, 16])

        >>> (out == 0.0).all(-1)
        tensor([[False, False, False,  True,  True],
               [False, False, False, False, False]])
    """

    def __init__(self, features: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__(features, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # type: ignore
        """preform layer norm on valid timesteps in x as specifed by mask.

        Args:
            x: a tensor of shape (B, T, ...)
            mask: boolean mask of of shape (B, T) where **valid** timesteps are True and
                invalid (padding) timesteps are false.
        """
        out = torch.zeros_like(x)
        out[mask] += F.layer_norm(x[mask], self.normalized_shape, self.weight, self.bias, self.eps)
        return out
