from typing import Dict, Sequence, List, Union

from hearth._file_utils import load_json, save_json
from hearth.text.utils import pad_tokens


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
