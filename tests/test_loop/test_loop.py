from typing import Dict

import numpy as np
import torch

from torch import nn
from hearth.metrics import BinaryAccuracy
from hearth.loop import Loop

from hearth.modules import BaseModule

from hearth.losses import MultiHeadLoss
from hearth.containers import TensorDict
from hearth.data.datasets import XYDataset, TensorDataset
from hearth.optimizers import AdamW


class TwoHeadedModel(BaseModule):
    def __init__(self, in_features=5, n_classes=4):
        super().__init__()
        self.class_head = nn.Linear(in_features, 4)
        self.binary_head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return dict(a=self.class_head(x), b=self.binary_head(x))


def gen_xor_dataset(n: int, val_split: float = 0.2):
    x = torch.rand(n, 2)
    y = np.logical_xor(x[:, 0] >= 0.5, x[:, 1] >= 0.5).unsqueeze(-1) * 1.0  # type: ignore
    x = x * 2 - 1

    n_train = n - int(round(n * val_split))

    train = TensorDataset(x[:n_train], y[:n_train])
    val = TensorDataset(x[n_train:], y[n_train:])

    return train, val


def test_full_loop_on_simple_xor():
    train, val = gen_xor_dataset(10000)

    train_batches = train.batches(shuffle=True, drop_last=False)
    val_batches = val.batches(batch_size=32, shuffle=True, drop_last=False)

    model = nn.Sequential(
        nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
    )

    loop = Loop(
        model=model,
        optimizer=AdamW(lr=0.001),
        loss_fn=nn.BCELoss(),
        metrics=BinaryAccuracy(),
    )

    loop(train_batches, val_batches, 4)
    assert loop.epoch == 4
    assert loop.stage == 'val'
    assert loop.metric >= 0.9


def test_multioutput(mocker):
    targets = TensorDict(a=torch.randint(4, size=(10,)), b=torch.rand(10, 1).round())
    inputs = torch.normal(0, 1, size=(10, 5))
    train = XYDataset(inputs[:8], targets[:8])

    val = XYDataset(inputs[-2:], targets[-2:])

    train_batches = train.batches(batch_size=8, shuffle=True, drop_last=False)
    val_batches = val.batches(batch_size=8, shuffle=True, drop_last=False)

    model = TwoHeadedModel()

    loop = Loop(
        model=model,
        optimizer=AdamW(lr=0.001),
        loss_fn=MultiHeadLoss(a=nn.CrossEntropyLoss(), b=nn.BCEWithLogitsLoss()),
    )

    assert loop._is_multihead_loss
    assert loop._loss_agg_key == 'weighted_sum'

    x, y = next(iter(train_batches))
    print(y)
    yhat = model(x)
    expected_loss = loop.loss_fn(yhat, y)
    loop.loss_fn.reset()

    spy = mocker.spy(loop, 'backward')
    loop(train_batches, val_batches, 1)

    spy.assert_called_once()
    backward_called_with = spy.call_args[0][0]
    assert isinstance(backward_called_with, torch.Tensor)
    torch.testing.assert_allclose(backward_called_with, expected_loss.weighted_sum)
    assert backward_called_with.grad_fn.name() == expected_loss.weighted_sum.grad_fn.name()
