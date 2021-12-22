from dataclasses import dataclass


@dataclass
class Event:
    """base class for all events."""

    @property
    def msg(self) -> str:
        return ''

    def logmsg(self) -> str:
        return f'{self.__class__.__name__}{self.msg}'


@dataclass
class MonitoringEvent(Event):
    """base class for monitoring events"""

    field: str
    stage: str
    steps: int

    def _get_stepmsg(self) -> str:
        return f'{self.steps} step{"s" if self.steps > 1 else ""}'

    def _get_fieldmsg(self) -> str:
        return f'{self.stage}.{self.field}'

    def logmsg(self) -> str:
        return f'{self.__class__.__name__}[{self._get_fieldmsg()}] {self.msg}'


@dataclass
class Improvement(MonitoringEvent):
    """This event should be emitted on an monitored improvement.

    Args:
        field: the field being monitored.
        stage: the stage being monitored.
        steps: the number of steps (generally epochs) that between this and the last improvement.
        best: the best value.
        last_best: the last best value
    """

    best: float
    last_best: float

    @property
    def msg(self) -> str:
        return (
            f'improved from : {self.last_best:0.4f} to {self.best:0.4f} in {self._get_stepmsg()}.'
        )


@dataclass
class Stagnation(MonitoringEvent):
    """This event should be emitted when a monitored metric or loss is not improving.

    Args:
        field: the field being monitored.
        stage: the stage being monitored.
        steps: the number of steps (generally epochs) that the measured value has been stagnant.
        best: the best value seen so far.
    """

    best: float

    @property
    def msg(self) -> str:
        return f'stagnant for {self._get_stepmsg()}' f' no improvement from {self.best:0.4f}.'


@dataclass
class CheckpointSaved(Event):
    """This event should be emitted when a model is saved as part of checkpointing in a callback.

    Args:
        model_dir: the directory the model was saved to.
    """

    model_dir: str

    @property
    def msg(self):
        return f' checkpoint saved to {self.model_dir}.'


@dataclass
class EarlyStop(Event):
    """this event should be emitted when training is instructed to stop early.

    Args:
        epoch: the epoch that stopping is triggered.
    """

    epoch: int

    @property
    def msg(self):
        return f' triggered at epoch {self.epoch}.'


@dataclass
class UnbottleEvent(Event):
    """this event should be emitted when fintuning.

    Args:
        epoch: the epoch that stopping is triggered.
    """

    epoch: int
    block: str
    lr: float

    @property
    def msg(self):
        return f' Unbottled block {self.block} with learning rate {self.lr} on epoch {self.epoch}.'


@dataclass
class UnbottlingComplete(Event):
    """this event should be emitted when unbottling is complete during finetuning.

    Args:
        epoch: the epoch that stopping is triggered.
    """

    @property
    def msg(self):
        return ' Unbottling complete nothing left to unbottle.'
