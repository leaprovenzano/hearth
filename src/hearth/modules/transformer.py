import torch
from torch import nn

from hearth.activations import get_activation
from hearth.modules.base import BaseModule
from hearth.modules.normalization import MaskedLayerNorm
from hearth.modules.wrappers import TimeMasked
from hearth.modules.attention import SelfAttention
from hearth.modules.embeddings import TransformerEmbedding


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


def _transform_positional_embeddings(positional_emb_weights, max_len):
    orig_len, feats = positional_emb_weights.shape
    if max_len == orig_len:
        # we need ot inject zero filled padding idx...
        return torch.cat([positional_emb_weights.new_zeros(1, feats), positional_emb_weights])
    return positional_emb_weights


class Bertish(BaseModule):
    """Bert style base transformer model, supports loading huggingface transformers weights via \
    `load_transformers_bert_state_dict` method.

    Args:
        features: number of base features. Defaults to 256.
        layers: number of transformer encoder layers. Defaults to 4.
        vocab_size: size of vocabulary. Defaults to 30522.
        attn_heads: number of attention heads. Defaults to 4.
        boom_scale: scale for intermediate size in feedforward network. Defaults to 4 (commonly
            used in bert archetectures).
        max_len: max sequence length (needed for positional embeddings). Defaults to 512.
        padding_idx: padding idx. Defaults to 0.
        layer_norm_eps: epsilon for layer norms used throught the network. Defaults to 1e-12.
        dropout: dropout rate for non-attention parts of network. Defaults to 0.1.
        attn_dropout: dropout rate for attention. Defaults to 0.1.
        activation: named activation for feedforward part of network. Defaults to 'gelu'.
        drop_head: If true drop attention from entire attention heads as dropout strategy otherwise
            drop timesteps from different heads. Defaults to False.
        pre_norm If true use pre-normalization strategy which may require less careful
            lr scheduling etc... Defaults to False as in bert-based models.
    """

    def __init__(
        self,
        features: int = 256,
        layers: int = 4,
        vocab_size: int = 30522,
        attn_heads: int = 4,
        boom_scale: int = 4,
        max_len: int = 512,
        padding_idx: int = 0,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        activation: str = 'gelu',
        drop_head: bool = False,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.out_features = self.features = features
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            features=features,
            max_len=max_len,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            padding_idx=padding_idx,
        )
        self.encoder = TransformerEncoder(
            features=features,
            layers=layers,
            n_heads=attn_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            boom_scale=boom_scale,
            drop_head=drop_head,
            pre_norm=pre_norm,
            layer_norm_eps=layer_norm_eps,
        )

    def blocks(self):
        yield self.embedding
        yield from self.encoder.blocks()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = self.embedding.build_mask(tokens)
        return self.encoder(self.embedding(tokens), mask)

    def load_transformers_bert_state_dict(self, sd):
        """load statedict from a hugging face transformers model"""
        new_sd = {
            'embedding.position_embeddings.embedding.weight': _transform_positional_embeddings(
                sd['embeddings.position_embeddings.weight'],
                self.embedding.position_embeddings.max_len,
            )
        }

        new_sd['embedding.word_embeddings.weight'] = sd['embeddings.word_embeddings.weight']
        if 'embeddings.token_type_embeddings.weight' in sd:
            new_sd['embedding.word_embeddings.weight'] += sd[
                'embeddings.token_type_embeddings.weight'
            ][0]

        frm_prefix = 'encoder.layer'
        to_prefix = 'encoder.layers'
        for kind in ['weight', 'bias']:

            new_sd[f'embedding.norm.{kind}'] = sd[f'embeddings.LayerNorm.{kind}']

            for i in range(self.encoder.depth()):

                # concat seperate qkv into one big one as we do in hearth
                # self attention impl....
                new_sd[f'{to_prefix}.{i}.attn.qkv.layer.{kind}'] = torch.cat(
                    [
                        sd[f'{frm_prefix}.{i}.attention.self.query.{kind}'],
                        sd[f'{frm_prefix}.{i}.attention.self.key.{kind}'],
                        sd[f'{frm_prefix}.{i}.attention.self.value.{kind}'],
                    ]
                )

                new_sd[f'{to_prefix}.{i}.attn_norm.{kind}'] = sd[
                    f'{frm_prefix}.{i}.attention.output.LayerNorm.{kind}'
                ]
                new_sd[f'{to_prefix}.{i}.attn.output.layer.{kind}'] = sd[
                    f'{frm_prefix}.{i}.attention.output.dense.{kind}'
                ]
                new_sd[f'{to_prefix}.{i}.ff_norm.{kind}'] = sd[
                    f'{frm_prefix}.{i}.output.LayerNorm.{kind}'
                ]
                new_sd[f'{to_prefix}.{i}.ff.layer.in_layer.{kind}'] = sd[
                    f'{frm_prefix}.{i}.intermediate.dense.{kind}'
                ]
                new_sd[f'{to_prefix}.{i}.ff.layer.out_layer.{kind}'] = sd[
                    f'{frm_prefix}.{i}.output.dense.{kind}'
                ]

        self.load_state_dict(new_sd)
        self.embedding.word_embeddings._fill_padding_idx_with_zero()
        self.embedding.position_embeddings.embedding._fill_padding_idx_with_zero()
