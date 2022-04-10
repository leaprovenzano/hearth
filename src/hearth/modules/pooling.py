import torch
from torch import Tensor, nn

from hearth.modules import BaseModule


class AttentionPool1D(BaseModule):
    """Attention pool for (batch first) timeseries data with masking.

    Args:
        in_features: number of input features.

    Example:
        >>> from hearth.modules import AttentionPool1D
        >>>
        >>> pool = AttentionPool1D(16)
        >>> mask = torch.tensor([[ True,  True,  True, False, False],
        ...                      [ True,  True,  True,  True,  True]])
        >>> inp = torch.rand(2, 5, 16)
        >>> out = pool(inp, mask)
        >>> out.shape
        torch.Size([2, 16])
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = self.out_features = in_features
        self.attn_weights = nn.Linear(self.in_features, 1)

    @torch.jit.export
    def attention(self, x: Tensor, mask: Tensor) -> Tensor:
        """get attention weights for the given input and mask.

        Args:
            x: (batch, timesteps, features) inputs
            mask: (batch, timesteps) mask where valid timesteps are True and padding timesteps
                are False.

        Returns:
            Tensor (batch, timesteps)
        """
        return self.attn_weights(x).squeeze(-1).masked_fill(~mask, -float('inf')).softmax(-1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """pool timesteps using attention.

        Args:
            x: (batch, timesteps, features) inputs
            mask: (batch, timesteps) mask where valid timesteps are True and padding timesteps
                are False.

        Returns:
            Tensor (batch, features)
        """
        return (x * self.attention(x, mask).unsqueeze(-1)).sum(1)
