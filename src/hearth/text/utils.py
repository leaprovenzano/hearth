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


def norm_whitespace(s: str) -> str:
    """normalize whitespace in the given string.

    Example:
        >>> from hearth.text.utils import norm_whitespace
        >>>
        >>> norm_whitespace('\tthere\t\tshould     only be  one   space  between  \twords.   ')
        'there should only be one space between words.'
    """
    return ' '.join(s.split())
