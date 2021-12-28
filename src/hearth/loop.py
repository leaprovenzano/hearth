from typing import Sequence, Callable, Optional, Union
import torch
from torch import nn
from contextlib import nullcontext
from hearth.callbacks import Callback, CallbackManager
from hearth.metrics import Running
from hearth.callbacks import History
from hearth.metrics import MetricStack
from hearth.losses import MultiHeadLoss
from hearth.optimizers import LazyOptimizer


class Loop:

    """The simplest kind of loop for basic supervised learning.

    Note:
        If you have more custom things you'd like to to that cant be handled
        in callbacks it's recommended to subclass this and overide the  ``handle_batch`` method.
    """

    stages = ('train', 'val')

    def __init__(
        self,
        model: nn.Module,
        optimizer: LazyOptimizer,
        loss_fn: Callable,
        metrics: Optional[Union[Callable, MetricStack, Sequence[Callable]]] = None,
        callbacks: Sequence[Callback] = (),
        history: Optional[History] = None,
        device: Union[torch.device, str] = 'cpu',
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.to(device)
        self.optimizer = optimizer

        self.metrics = metrics
        self.n_batches = 0
        self.batches_seen = 0
        self.stage = self.stages[0]
        self.should_stop = False
        self.history = history if history is not None else History()
        self.callbacks = CallbackManager(self.history, *callbacks)
        self.callbacks.on_registration(self)
        self.epoch = self.history.last_epoch + 1 if len(self.history) else 0

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if isinstance(optimizer, LazyOptimizer) and not optimizer.initialized:
            optimizer.add_model(self.model)
        self._optimizer = optimizer

    def to(self, device: Union[torch.device, str]):
        self.device = device
        self.model.to(self.device)
        self._loss_fn.to(self.device)

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Optional[Union[Callable, MetricStack, Sequence[Callable]]] = None):
        if metrics:
            if not isinstance(metrics, MetricStack):
                if isinstance(metrics, dict):
                    metrics = MetricStack(**metrics)
                elif isinstance(metrics, Sequence):
                    metrics = MetricStack(*metrics)
                else:
                    metrics = MetricStack(metrics)

            self._metrics = Running(metrics)
            self._has_metrics = True
        else:
            self._metrics = Running(MetricStack())
            self._has_metrics = False

    @property
    def loss_fn(self):
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, loss_fn=Callable):
        self._is_multihead_loss = isinstance(loss_fn, MultiHeadLoss)
        self._loss_agg_key = loss_fn.aggregate_key if self._is_multihead_loss else None
        self._loss_fn = Running(loss_fn)

    @property
    def loss(self):
        return self.loss_fn.average

    @property
    def metric(self):
        if self._has_metrics:
            return self.metrics.average
        return None

    def grad_context(self):
        if self.stage == 'train':
            return nullcontext()
        return torch.no_grad()

    def _requires_backward(self) -> bool:
        return self.stage == 'train'

    def optimizer_step(self):
        self.optimizer.step()

    def _optimizer_step(self):
        self.callbacks.on_step_start(self)
        self.optimizer_step()
        self.callbacks.on_step_end(self)

    def compute_loss(self, yhat, ytru, **kwargs):
        return self.loss_fn(yhat, ytru, **kwargs)

    def _compute_loss(self, yhat, ytru, **kwargs):
        self.callbacks.on_loss_start(self)
        loss = self.compute_loss(yhat, ytru, **kwargs)
        self.callbacks.on_loss_end(self)
        if self._is_multihead_loss:
            return loss[self._loss_agg_key]
        return loss

    def compute_metric(self, yhat, ytru, **kwargs):
        return self.metrics(yhat, ytru, **kwargs)

    def _compute_metric(self, yhat, ytru, **kwargs):
        with torch.no_grad():
            self.callbacks.on_metric_start(self)
            metric = self.compute_metric(yhat, ytru, **kwargs)
            self.callbacks.on_metric_end(self)
        return metric

    def _backward(self, loss, **kwargs):
        self.callbacks.on_backward_start(self)
        self.backward(loss)
        self.callbacks.on_backward_end(self)

    def backward(self, loss, **kwargs):
        loss.backward()

    def _forward(self, x, **kwargs):
        with self.grad_context():
            return self.forward(x, **kwargs)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def handle_batch(self, batch):
        # do forward pass and get loss
        self.optimizer.zero_grad()
        # unpack the batch... you can override this if your batch differs...
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self._forward(x)
        loss = self._compute_loss(y_hat, y)
        if self._requires_backward():
            self._backward(loss)
            self._optimizer_step()
        if self._has_metrics:
            self._compute_metric(y_hat, y)

    def handle_batches(self, batches):
        self.n_batches = len(batches)
        self.batches_seen = 0
        for batch in batches:
            self.callbacks.on_batch_start(self)
            self.handle_batch(batch)
            self.batches_seen += 1
            self.callbacks.on_batch_end(self)

    def handle_stage(self, stage, batches):
        self.stage = stage
        if self.stage == 'train':
            self.model.train()
        else:
            self.model.eval()
        if self._has_metrics:
            self.metrics.reset()
        self.loss_fn.reset()
        self.callbacks.on_stage_start(self)
        self.handle_batches(batches)
        self.callbacks.on_stage_end(self)

    def fire(self, event):
        self.callbacks.on_event(self, event)

    def __call__(self, train, val, epochs: int = 1):
        self.should_stop = False
        for _ in range(epochs):
            self.callbacks.on_epoch_start(self)
            for stage, batches in zip(self.stages, (train, val)):
                self.handle_stage(stage, batches)
            self.callbacks.on_epoch_end(self)
            self.epoch += 1
            if self.should_stop:
                break

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(model={self.model},'
            f' optimizer={self.optimizer}'
            f' loss_fn={self.loss_fn!r}'
            f' metrics={self.metrics!r}'
            f' callbacks={self.callbacks})'
        )
