import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal
from dataclasses import dataclass
import operator
from functools import reduce, partial
from hearth.callbacks.base import Callback
from hearth.events import Improvement, Stagnation


def dotted_attrgetter(path: str, obj):
    return reduce(getattr, path.split('.'), obj)


@dataclass
class ImprovementMonitor(Callback):

    field: str = 'loss'
    improvement_on: Literal['gt', 'lt'] = 'lt'
    stage: str = 'val'
    stagnant_after: int = 1

    def __post_init__(self):
        if self.improvement_on not in ('ge', 'lt'):
            raise ValueError(
                f'improvement_on must be one of ["ge", "lt"] but got {self.improvement_on}'
            )
        self._get_value = partial(dotted_attrgetter(self.field))
        self._is_improvement = getattr(operator, self.improvement_on)
        self._last_best = float('inf') if self.improvement_on == 'lt' else -float('inf')
        self._best_step = -1

    def on_stage_end(self, loop):
        if loop.stage == self.stage:
            this_value = self._get_value(loop)
            steps = loop.epoch - self._best_step
            event = None
            if self._is_improvement(this_value, self._last_best):
                event = Improvement(
                    self.field, steps=steps, best=this_value, last_best=self._last_best
                )
                self._best_step = loop.epoch
                self._last_best = this_value
            elif steps > self.stagnant_after:
                event = Stagnation(field=self.field, steps=steps, best=self._last_best)
            if event:
                loop.fire(event)