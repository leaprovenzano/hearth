from dataclasses import dataclass

from hearth.callbacks import Callback
from hearth.modules import BaseModule
from hearth.optimizers import LazyOptimizer
from hearth.events import UnbottleEvent, UnbottlingComplete


@dataclass
class FineTuneCallback(Callback):
    """This callback gradually unbottles blocks and adds them to the optimizer decaying learning \
    rate by depth.

    Note:
        only models derived from :class:`hearth.modules.BaseModule` and optimizers derived
        from :class:`hearth.modules.LazyOptimizer` currently supported.

    Args:
        start_epoch: start unbottling blocks at this epoch.
        unbottle_every: number of epochs to wait between unbottles.
        decay: lr for each block will be base_lr/ (decay^depth). Defaults to 2.6 (as in ULMfit).
        max_depth: maximum depth (in blocks) to unbottle. If -1 then keep unbottling until
            all blocks are trainable. Defaults to -1.

    **Active On:**
        - registration
        - epoch_start

    **Events Emitted:**
        - :class:`hearth.events.UnbottleEvent`

    **Accesses Loop Attributes:**
        - model
        - optimizer
    """

    start_epoch: int
    unbottle_every: int = 1
    decay: float = 2.6
    max_depth: int = -1

    def __post_init__(self):
        self._unbottling_complete: bool = False
        self._depth = 0

    def on_registration(self, loop):
        # check model and optimizer type
        if not isinstance(loop.model, BaseModule):
            raise TypeError(
                f'{self.__class__.__name__} only supports hearth.BaseModule subclasses.'
            )
        if not isinstance(loop.optimizer, LazyOptimizer):
            raise TypeError(
                f'{self.__class__.__name__} only supports hearth.LazyOptimizer subclasses.'
            )

    def _should_unbottle(self, epoch: int) -> bool:
        if not self._unbottling_complete:
            return (
                epoch >= self.start_epoch and (self.start_epoch + epoch) % self.unbottle_every == 0
            )
        return False

    def _get_lr(self, optimizer):
        # get base lr
        lr = optimizer.get_lrs()[0]
        return lr / (self.decay ** self._depth)

    def _unbottle(self, loop):
        next_block = loop.model.unbottle()
        # if theres nothing left this will be None and
        # so unbottling is complete
        if not next_block:
            self._unbottling_complete = True
            return UnbottlingComplete()
        # otherwise unbottle
        self._depth += 1
        lr = self._get_lr(loop.optimizer)
        loop.optimizer.add_model(next_block, lr=lr)
        return UnbottleEvent(epoch=loop.epoch, block=next_block.__class__.__name__, lr=lr)

    def on_epoch_start(self, loop):
        if self._should_unbottle(loop.epoch):
            event = self._unbottle(loop)
            loop.fire(event)
            if (self.max_depth > -1) and (self._depth >= self.max_depth):
                self._unbottling_complete = True
                if not isinstance(event, UnbottlingComplete):
                    loop.fire(UnbottlingComplete())
