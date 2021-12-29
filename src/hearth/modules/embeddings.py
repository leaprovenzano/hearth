import torch
from torch import nn

from hearth.modules import BaseModule


class AbsolutePositionalEmbedding(BaseModule):
    """Absolute learned positional embeddings a la bert.

    Args:
        features: number of embedding features.
        max_len: max sequence length.
        padding_idx: used to mask padding timesteps

    Example:
        >>> from hearth.modules import AbsolutePositionalEmbedding
        >>>
        >>> emb = AbsolutePositionalEmbedding(256, max_len=512)
        >>> tokens = torch.tensor([[99, 6, 55, 1, 0, 0],
        ...                        [8, 22, 7, 8, 3, 11]])
        >>> out = emb(tokens)
        >>> out.shape
        torch.Size([2, 6, 256])

        >>> (out[tokens == 0] == 0).all()
        tensor(True)
    """

    def __init__(self, features: int, max_len: int = 512, padding_idx: int = 0):
        super().__init__()
        self.out_features = features
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(self.max_len + 1, features, padding_idx=self.padding_idx)
        self.register_buffer(
            "position_ids", torch.arange(1, self.max_len + 1).expand((1, -1)), persistent=False
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        timesteps = tokens.size(1)

        position_ids = self.position_ids[:, :timesteps]  # type: ignore
        position_ids = position_ids.expand_as(tokens).masked_fill(
            (tokens == self.padding_idx), 0
        )  # (bs, max_seq_length)

        return self.embedding(position_ids)
