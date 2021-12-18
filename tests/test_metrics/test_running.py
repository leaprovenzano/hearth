import pytest
import torch
from torch import nn
from hearth.metrics import Running, BinaryAccuracy


@pytest.mark.parametrize('running_fn, ', [Running(BinaryAccuracy()), Running(nn.BCELoss())])
def test_running_wrapper(running_fn):
    batches = [
        (torch.rand(5, 1), torch.randint(1, size=(5, 1)) * 1.0),
        (torch.rand(5, 1), torch.randint(1, size=(5, 1)) * 1.0),
        (torch.rand(3, 1), torch.randint(1, size=(3, 1)) * 1.0),
    ]

    for pred, target in batches:
        out = running_fn(pred, target)
        assert isinstance(out, torch.Tensor)

    assert running_fn._samples_seen == 13
    assert running_fn._batches_seen == 3

    full_preds = torch.cat([b[0] for b in batches])
    full_targets = torch.cat([b[1] for b in batches])
    assert running_fn.average == pytest.approx(running_fn.fn(full_preds, full_targets).item())

    # test_reset
    running_fn.reset()
    assert running_fn._samples_seen == 0
    assert running_fn._batches_seen == 0
    assert running_fn.average == 0


def test_grad():
    yhat, target = torch.rand(5, 1, requires_grad=True), torch.randint(1, size=(5, 1)) * 1.0
    running_loss = Running(nn.BCELoss())
    out = running_loss(yhat, target)
    assert out.grad_fn.name().startswith('BinaryCrossEntropyBackward')
