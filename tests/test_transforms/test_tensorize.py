from hearth.data.transforms import Tensorize
import torch
import numpy as np
import pytest


def test_bad_dtype():
    expect_msg = 'dtype must be a valid torch.dtype or a string .+ got crap.'
    with pytest.raises(TypeError, match=expect_msg):
        Tensorize(dtype='crap')


@pytest.mark.parametrize(
    'inp, expected_dtype',
    [
        ([1, 2, 3], torch.int64),
        (np.array([1, 2, 3], dtype='float32'), torch.float32),
        (np.array([1, 2, 3], dtype='float64'), torch.float64),
        ([True, True, False], torch.bool),
    ],
)
def test_none_dtype(inp, expected_dtype):
    transform = Tensorize()
    out = transform(inp)
    assert out.dtype == expected_dtype


@pytest.mark.parametrize('dtype,', ['float32', torch.float32])
def test_set_dtype(dtype):
    transform = Tensorize(dtype=dtype)
    assert transform._dtype == torch.float32
    assert transform([1, 2, 3]).dtype == torch.float32
