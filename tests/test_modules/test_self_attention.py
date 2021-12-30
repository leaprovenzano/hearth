import pytest
from hearth.modules import SelfAttention


def test_invalid_n_heads():
    with pytest.raises(ValueError, match='out_features must be divisible by n_heads'):
        SelfAttention(35, n_heads=4)
