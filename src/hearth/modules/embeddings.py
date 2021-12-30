import torch
from torch import nn

from hearth.modules import BaseModule
from hearth.modules.normalization import MaskedLayerNorm


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
            'position_ids', torch.arange(1, self.max_len + 1).expand((1, -1)), persistent=False
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        timesteps = tokens.size(1)

        position_ids = self.position_ids[:, :timesteps]  # type: ignore
        position_ids = position_ids.expand_as(tokens).masked_fill(
            (tokens == self.padding_idx), 0
        )  # (bs, max_seq_length)

        return self.embedding(position_ids)


class TransformerEmbedding(BaseModule):
    """simple version of embedding used in transformer models.

    Note:
      this embedding does not use token type embeddings as used in some bert models. If
      using with a pretrained model it's reccomended that you first add these to word embeddings.

    Args:
        vocab_size: vocabulary size for word embeddings.
        features: number of features in embedding space.
        max_len: maximum sequence length for positional embeddings. Defaults to 512.
        dropout: dropout rate. Defaults to 0.1.
        layer_norm_eps: epsilon for layer normalization. Defaults to 1e-12.
        padding_idx: index for non-valid padding timesteps. Defaults to 0.

    Example:
        >>> from torch import nn
        >>> from hearth.modules import TransformerEmbedding
        >>>
        >>> emb = TransformerEmbedding(1000, 256, padding_idx=0)

        >>> tokens = torch.tensor([[99, 6, 55, 1, 0, 0],
        ...                        [8, 22, 7, 8, 3, 11]])
        >>> out = emb(tokens)
        >>> out.shape
        torch.Size([2, 6, 256])

        >>> (out == 0.0).all(-1)
        tensor([[False, False, False, False, True,  True],
                [False, False, False, False, False, False]])

        >>> emb.build_mask(tokens)
        tensor([[ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True]])
    """

    def __init__(
        self,
        vocab_size: int,
        features: int,
        max_len: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.out_features = features
        self.padding_idx = padding_idx
        self.word_embeddings = nn.Embedding(vocab_size, features, padding_idx=padding_idx)
        self.position_embeddings = AbsolutePositionalEmbedding(
            features, max_len, padding_idx=padding_idx
        )
        self.norm = MaskedLayerNorm(features, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    @torch.jit.export
    def build_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """get a mask where all valid timesteps are True"""
        return tokens != self.padding_idx

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.word_embeddings(tokens) + self.position_embeddings(tokens)
        x = self.norm(x, self.build_mask(tokens))
        return self.dropout(x)
