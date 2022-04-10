from typing import List


def pad_tokens(tokens: List[List[int]], pad_value: int = 0) -> List[List[int]]:
    """pad a batch of tokens to fixed maximum lengh using `pad_value`.

    Args:
        tokens: list of list of tokens of varying lengths.
        pad_value: padding value. Defaults to 0.

    Example:
        >>> from hearth.text.utils import pad_tokens
        >>>
        >>> tokens = [[1, 2], [1, 2, 3], [1]]
        >>> pad_tokens(tokens)
        [[1, 2, 0], [1, 2, 3], [1, 0, 0]]
    """
    maxlen = max(map(len, tokens))
    return [seq + [pad_value] * (maxlen - len(seq)) for seq in tokens]
