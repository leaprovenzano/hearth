import pytest
import torch
from torch import nn
from hearth.modules import BaseModule


class Dummy(BaseModule):
    def __init__(self, in_feats: int, hidden: int, out_feats: int):
        super().__init__()
        self.lin1 = nn.Linear(in_feats, hidden)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(hidden, out_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


def test_freeze():
    model = Dummy(4, 5, 2)
    model.freeze()
    for param in model.parameters():
        assert not param.requires_grad


def test_unfreeze():
    model = Dummy(4, 5, 2)
    model.freeze()
    model.unfreeze()
    for param in model.parameters():
        assert param.requires_grad


def test_script():
    model = Dummy(4, 5, 2)
    model.freeze()

    try:
        scripted = model.script()
    except RuntimeError as err:
        pytest.fail(f'failed to script model with script method! raised {err}')

    x = torch.normal(0, 1, (3, 4))
    model_y = model(x)
    scripted_y = scripted(x)
    torch.testing.assert_allclose(scripted_y, model_y)
    assert scripted_y.requires_grad == model_y.requires_grad