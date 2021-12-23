from dataclasses import dataclass
import re
from typing import Union

import torch
from torch import nn

import pytest

from hearth.optimizers import LazyOptimizer
from hearth.modules import BaseModule
from hearth.callbacks import FineTuneCallback
from hearth.optimizers import AdamW
from hearth.events import UnbottleEvent, UnbottlingComplete


@dataclass
class DummyLoop:

    model: nn.Module
    optimizer: Union[torch.optim.Optimizer, LazyOptimizer]
    callback: FineTuneCallback
    epoch: int = 0

    def __post_init__(self):
        self._event_log = []
        self._setup_optimizer()

    def _setup_optimizer(self):
        if isinstance(self.optimizer, LazyOptimizer):
            self.optimizer.add_model(self.model)

    def fire(self, event):
        self._event_log.append(event)

    def on_registration(self):
        self.callback.on_registration(self)

    def on_epoch_start(self):
        self.callback.on_epoch_start(self)

    def do_epoch(self):
        self.on_epoch_start()
        self.epoch += 1

    def run(self, epochs):
        for _ in range(epochs):
            self.do_epoch()


class HearthModel(BaseModule):
    def __init__(self, in_feats: int = 3, hidden: int = 16, out_feats: int = 6):
        super().__init__()
        self.in_transform = nn.Linear(in_feats, out_feats)
        self.hidden = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
        )
        self.out = nn.Linear(hidden, out_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(self.in_transform(x)))

    def blocks(self):
        yield self.in_transform
        yield from self.hidden
        yield self.out


@pytest.mark.parametrize(
    'loop, msg',
    [
        (
            DummyLoop(model=nn.Linear(5, 5), optimizer=AdamW(), callback=FineTuneCallback(2)),
            'FineTuneCallback only supports hearth.BaseModule subclasses.',
        ),
        (
            DummyLoop(
                model=HearthModel(),
                optimizer=torch.optim.AdamW(HearthModel().parameters()),
                callback=FineTuneCallback(2),
            ),
            'FineTuneCallback only supports hearth.LazyOptimizer subclasses.',
        ),
    ],
)
def test_invalid_at_registration(loop, msg):
    with pytest.raises(TypeError, match=re.escape(msg)):
        loop.on_registration()


@pytest.mark.parametrize(
    'callback, epochs,',
    [
        (FineTuneCallback(start_epoch=3, unbottle_every=1), [3, 4, 5]),
        (FineTuneCallback(start_epoch=3, unbottle_every=2), [3, 5, 7]),
    ],
)
def test_should_unbottle(callback, epochs):
    for epoch in epochs:
        assert callback._should_unbottle(epoch)


@pytest.mark.parametrize(
    'callback, epochs,',
    [
        (FineTuneCallback(start_epoch=3, unbottle_every=1), list(range(3))),
        (FineTuneCallback(start_epoch=3, unbottle_every=2), list(range(3)) + [4, 6, 8]),
    ],
)
def test_should_not_unbottle(callback, epochs):
    for epoch in epochs:
        assert not callback._should_unbottle(epoch)


def test_events_emitted():
    base_lr = 1.0
    callback = FineTuneCallback(start_epoch=3, unbottle_every=2)
    model = HearthModel()
    model.bottleneck(3)
    loop = DummyLoop(model=model, optimizer=AdamW(lr=1.0), callback=callback)
    expected_events = [
        UnbottleEvent(epoch=3, block='Linear', lr=base_lr / callback.decay),
        UnbottleEvent(epoch=5, block='Linear', lr=base_lr / (callback.decay ** 2)),
        UnbottleEvent(epoch=7, block='Linear', lr=base_lr / (callback.decay ** 3)),
        UnbottlingComplete(),
    ]

    loop.run(11)
    assert loop._event_log == expected_events


def test_events_emitted_with_max_depth():
    base_lr = 1.0
    callback = FineTuneCallback(start_epoch=3, unbottle_every=2, max_depth=2)
    model = HearthModel()
    model.bottleneck(3)
    loop = DummyLoop(model=model, optimizer=AdamW(lr=1.0), callback=callback)
    expected_events = [
        UnbottleEvent(epoch=3, block='Linear', lr=base_lr / callback.decay),
        UnbottleEvent(epoch=5, block='Linear', lr=base_lr / (callback.decay ** 2)),
        UnbottlingComplete(),
    ]

    loop.run(11)
    assert loop._event_log == expected_events
