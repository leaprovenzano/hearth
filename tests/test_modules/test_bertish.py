import pytest
import torch
from hearth.modules import Bertish
from tests.utils import random_padded_tokens


@pytest.mark.parametrize('model,', [Bertish()])
def test_torchscriptability(model):
    tokens = random_padded_tokens(5)
    model.eval()
    with torch.no_grad():
        pyout = model(tokens)

    try:
        scripted = model.script()
        with torch.no_grad():
            scripted_out = scripted(tokens)
    except RuntimeError as err:
        pytest.fail(f'failed to script model with script method! raised {err}')
    assert scripted_out.shape == pyout.shape
    torch.testing.assert_allclose(scripted_out, pyout)
