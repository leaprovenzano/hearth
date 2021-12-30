import torch
from torch import nn

from hearth.activations import get_activation
from hearth.modules.base import BaseModule
from hearth.modules.normalization import MaskedLayerNorm
from hearth.modules.wrappers import TimeMasked
from hearth.modules.attention import SelfAttention


class Boom(BaseModule):
    """feedforward part of std transformer network sometimes affectionatly referred to as the\
    bOOm layer (since it expands and contracts).


    Args:
        in_features: number of input features.
        scale: scale for intermediate size (the OO in bOOm). Defaults to 4 (commonly used in
            bert archetectures.)
        activation: named activation. Defaults to 'gelu'.
        dropout: Dropout rate for intermediate activation. Defaults to 0.1.

    Example:
        >>> import torch
        >>> from hearth.modules import Boom
        >>>
        >>> layer = Boom(16, scale=4, activation='gelu', dropout=0.1)
        >>> inp = torch.rand(10, 16)
        >>> layer(inp).shape
        torch.Size([10, 16])
    """

    def __init__(
        self, in_features: int, scale: int = 4, activation: str = 'gelu', dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = self.out_features = in_features
        self.hidden_feats = self.in_features * scale
        self.in_layer = nn.Linear(self.in_features, self.hidden_feats)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.hidden_feats, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.out_layer(x)


class TransformerEncoderBlock(BaseModule):
    """a single transformer encoder block.

    Args:
        features: number of features.
        n_heads: number of attention heads.
        dropout: general dropout rate used in feedforward part of network and between
            residual connections. Defaults to 0.1.
        attn_dropout: dropout for self attention. Defaults to 0.1.
        activation: named activation for feedforward part of network. Defaults to 'gelu'.
        boom_scale: scale for intermediate size in feedforward network. Defaults to 4 (commonly
            used in bert archetectures).
        drop_head If true drop attention from entire attention heads as dropout strategy otherwise
            drop timesteps from different heads. Defaults to False.
        pre_norm: If true use pre-normalization strategy which may require less careful
            lr scheduling etc... Defaults to False as in bert-based models.
        layer_norm_eps: epsilon for layer norms used throught the network. Defaults to 1e-12.

    Example:
        >>> import torch
        >>> from hearth.modules import TransformerEncoderBlock
        >>>
        >>> layer = TransformerEncoderBlock(16, n_heads=4, boom_scale=4)
        >>> mask = torch.tensor([[ True,  True,  True, False, False],
        ...                      [ True,  True,  True,  True,  True]])
        >>> inp = torch.rand(2, 5, 16)
        >>> layer(inp, mask).shape
        torch.Size([2, 5, 16])

        with pre-norm scheme:

        >>> layer = TransformerEncoderBlock(16, n_heads=4, boom_scale=4, pre_norm=True)
        >>> layer(inp, mask).shape
        torch.Size([2, 5, 16])
    """

    def __init__(
        self,
        features: int,
        n_heads: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        activation: str = 'gelu',
        boom_scale: int = 4,
        drop_head: bool = False,
        pre_norm: bool = False,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.in_features = self.out_features = features
        self.attn_norm = MaskedLayerNorm(features, eps=layer_norm_eps)
        self.attn = SelfAttention(
            features, n_heads=n_heads, dropout=attn_dropout, drop_head=drop_head
        )
        self.attn_output_drop = nn.Dropout(dropout)

        self.ff_norm = MaskedLayerNorm(features, eps=layer_norm_eps)
        self.ff = TimeMasked(
            Boom(features, scale=boom_scale, dropout=dropout, activation=activation)
        )
        self.ff_output_drop = nn.Dropout(dropout)

        self.pre_norm = pre_norm

    def _sa_block(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.attn_output_drop(self.attn(x, mask))

    def _ff_block(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.ff_output_drop(self.ff(x, mask))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self._sa_block(self.attn_norm(x, mask), mask)
            x = x + self._ff_block(self.ff_norm(x, mask), mask)
        else:
            x = self.attn_norm(x + self._sa_block(x, mask), mask)
            x = self.ff_norm(x + self._ff_block(x, mask), mask)

        return x


class TransformerEncoder(BaseModule):

    """a stack of transformer encoder layers

    Args:
        features: number of features.
        layers: number of layers.
        n_heads: number of attention heads.
        dropout: general dropout rate used in feedforward part of network and between
            residual connections. Defaults to 0.1.
        attn_dropout: dropout for self attention. Defaults to 0.1.
        activation: named activation for feedforward part of network. Defaults to 'gelu'.
        boom_scale: scale for intermediate size in feedforward network. Defaults to 4 (commonly
            used in bert archetectures).
        drop_head If true drop attention from entire attention heads as dropout strategy otherwise
            drop timesteps from different heads. Defaults to False.
        pre_norm: If true use pre-normalization strategy which may require less careful
            lr scheduling etc... Defaults to False as in bert-based models.
        layer_norm_eps: epsilon for layer norms used throught the network. Defaults to 1e-12.

    Example:
        >>> import torch
        >>> from hearth.modules import TransformerEncoder
        >>>
        >>> model = TransformerEncoder(16, layers=3, n_heads=4, boom_scale=4)
        >>> mask = torch.tensor([[ True,  True,  True, False, False],
        ...                      [ True,  True,  True,  True,  True]])
        >>> inp = torch.rand(2, 5, 16)
        >>> model(inp, mask).shape
        torch.Size([2, 5, 16])

        >>> model.depth()
        3

    """

    def __init__(
        self,
        features: int,
        layers: int,
        n_heads: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        activation: str = 'gelu',
        boom_scale: int = 4,
        drop_head: bool = False,
        pre_norm: bool = False,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    features=features,
                    n_heads=n_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    activation=activation,
                    boom_scale=boom_scale,
                    drop_head=drop_head,
                    pre_norm=pre_norm,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(layers)
            ]
        )

    def blocks(self):
        yield from self.layers

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for mod in self.layers:
            x = mod(x, mask)
        return x
