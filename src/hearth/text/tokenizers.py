from typing import Dict, Sequence, List, Union

from hearth._file_utils import load_json, save_json
from hearth.text.utils import pad_tokens, norm_whitespace


class Tokenizer:
    """Base class for tokenizers implementing a few useful abstractions for seralization and batch\
    /non-batch usage.

    Not meant to be used directly. All child tokenizers should override the `tokenize` method at
    minimum.
    """

    def __init__(self, vocab: Dict[str, int], **kwargs):
        self.vocab = vocab

    def tokenize(self, s: str) -> List[int]:
        return NotImplemented

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def config(self):
        """get jsonable config dict for this Tokenizer.

        Used in combination with load and save for tokenizer serialization.
        """
        return {'vocab': self.vocab}

    @classmethod
    def from_config(cls, config):
        """load a config dictionary into a new instance of this tokenizer class"""
        return cls(**config)

    def save(self, path: str):
        """Save this tokenizer's config to as a json file at `path`."""
        save_json(self.config(), path)

    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        """load a new instance of this Tokenizer class from using config found at `path`."""
        config = load_json(path)
        return cls.from_config(config)

    def tokenize_batch(self, batch: Sequence[str]) -> List[List[int]]:
        """tokenize a batch of strings."""
        return pad_tokens(list(map(self.tokenize, batch)))

    def __call__(self, inp: Union[str, Sequence[str]]):
        """tokenize inputs, works with batch or single string inputs."""
        if isinstance(inp, str):
            return self.tokenize(inp)
        return self.tokenize_batch(inp)


class WordPieceTokenizer(Tokenizer):
    """Bert style word piece tokenizer.

    Args:
        vocab: Vocab dictionary mapping text to index.
        subword_prefix: defines how subwords are marked in the vocab. Defaults to '##' as in Bert.
        bos: token to prepend to the begining of the sequence. Defaults to '[CLS]' as in Bert.
        eos: token to prepend to the end of the sequence. Defaults to '[SEP]' as in Bert.
        oov: Token for handling out of vocabulary stuff. With word piece tokenization this tends
            to happen very rarely and usually indicates vocab characters as opposed to whole words
            but for completeness we add both a OOV and ##OOV to the vocabulary replacing the
            `[unused0]` and `[unused1]` tokens if no oov tokens matching this value are found in
            the vocab. Note that this differs a little from most Bert tokenizer implementations
            which do not acknowlege OOV. Defaults to 'OOV', consider changing if tokenizer is not
            lowercase.
        lower: bool If True lowercase everything. Default to True.

    """

    def __init__(
        self,
        vocab: Dict[str, int],
        subword_prefix: str = '##',
        bos: str = '[CLS]',
        eos: str = '[SEP]',
        oov: str = 'OOV',
        lower: bool = True,
    ):
        super().__init__(vocab)
        self.oov = oov
        if self.oov not in self.vocab:
            # replace [unused0] and [unused1] with OOV and subword OOV
            # if they are not already in vocab....
            self.vocab[self.oov] = self.vocab.pop('[unused0]')
            self.vocab[f'{subword_prefix}{self.oov}'] = self.vocab.pop('[unused1]')

        self.lower = lower
        self.bos = bos
        self.eos = eos
        self.subword_prefix = subword_prefix
        self.bos_idx = self.vocab[self.bos]
        self.eos_idx = self.vocab[self.eos]

    def clean(self, s: str):
        if self.lower:
            s = s.lower()
        return norm_whitespace(s)

    def config(self):
        config = super().config()
        config['subword_prefix'] = self.subword_prefix
        config['lower'] = self.lower
        config['bos'] = self.bos_token
        config['eos'] = self.eos_token
        config['oov'] = self.oov
        return config

    def subwords(self, s, _prefix=''):
        if _prefix + s in self.vocab:
            return [_prefix + s]

        for i in range(len(s) - 1, 0, -1):
            subw = f'{_prefix}{s[:i]}'
            if subw in self.vocab:
                return [subw] + self.subwords(s[i:], _prefix=self.subword_prefix)
        return [f'{_prefix}{self.oov}']

    def split(self, s: str):
        for w in s.split():
            if w in self.vocab:
                yield w
            else:
                yield from self.subwords(w)

    def tokenize(self, s):
        return [self.bos_idx] + list(map(self.vocab.__getitem__, self.split(s))) + [self.eos_idx]
