from abc import ABC, abstractclassmethod
from copy import deepcopy
from functools import singledispatch
import inspect
from inspect import Parameter
from torch import nn
import functools
import importlib

_global_param_exclude = ('device', 'dtype')


class SupportsConfig(ABC):
    """protocoll for classes that support a set of methods for load and save from json config."""

    @classmethod
    def from_config(cls, config):
        """load an instance of the class from a config dict."""
        return from_config(cls, config)

    @abstractclassmethod
    def config(self):
        """this method should return a **jsonable** dictionary that can be used to initilize this\
         class"""
        return {}

    @classmethod
    def __subclasshook__(cls, other):
        if cls is SupportsConfig:
            try:
                return (
                    hasattr(cls, 'config')
                    and callable(cls.config)
                    and hasattr(cls, 'from_config')
                    and callable(cls.from_config)
                )
            except AttributeError:
                return False
        return NotImplemented


def import_cls(name: str, module: str, **kwargs):
    return getattr(importlib.import_module(module), name)


def _parse_args(args, signature):  # noqa: C901
    def parse_arg(arg):
        if isinstance(arg, nn.Module):
            return extended_config(arg)
        return arg

    for name, arg in args:
        sig_param = signature.parameters[name]
        if sig_param.kind == Parameter.VAR_POSITIONAL:
            yield f'*{name}', [parse_arg(a) for a in arg]

        elif sig_param.kind == Parameter.VAR_KEYWORD:
            yield from ((k, parse_arg(v)) for k, v in arg.items())

        else:
            yield name, parse_arg(arg)


def _init_wrapper(init_fn):
    def init(self, *args, **kwargs):
        sig = inspect.signature(self.__class__)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        if not hasattr(self, '__config__'):
            self.__config__ = {}
            for k, v in _parse_args(bound.arguments.items(), sig):
                if k not in self.__config__:
                    self.__config__[k] = v

        init_fn(self, *args, **kwargs)

    init_ = functools.wraps(init_fn)(init)
    return init_


def from_config(cls, config):
    def load_item(x):
        if is_extended_config(x):
            return from_extended_config(x)
        return x

    config = deepcopy(config)
    varargs = list(filter(lambda x: x.startswith('*'), config))
    if varargs:
        args = config.pop(varargs[0])
    else:
        args = []
    return cls(*map(load_item, args), **{k: load_item(v) for k, v in config.items()})


def from_extended_config(extended_config):
    if not is_extended_config(extended_config):
        raise ValueError('config does not contain nessisary metadata')
    cls = import_cls(**extended_config['meta'])

    return from_config(cls, extended_config['config'])


def get_meta(cls):
    return {'name': cls.__name__, 'module': cls.__module__}


def is_extended_config(config) -> bool:
    return (
        isinstance(config, dict)
        and ('meta' in config)
        and ('name' in config['meta'])
        and ('module' in config['meta'])
    )


def gather_params_from_instance(inst, params=None, exclude=()):
    if params is None:
        params = inspect.signature(inst.__class__).parameters
    return {
        param: getattr(inst, param)
        for param in params
        if param not in _global_param_exclude + exclude
    }


def extended_config(obj):
    meta = get_meta(obj.__class__)
    conf = config(obj)
    return {'meta': meta, 'config': conf}


@singledispatch
def config(obj):
    if hasattr(obj, 'config'):
        return obj.config()
    return NotImplemented


@config.register(nn.Module)
def module_config(mod):
    if hasattr(mod, 'config'):
        return mod.config()
    return gather_params_from_instance(mod)


@config.register(nn.Bilinear)
@config.register(nn.Conv1d)
@config.register(nn.Conv2d)
@config.register(nn.Conv3d)
@config.register(nn.ConvTranspose1d)
@config.register(nn.ConvTranspose2d)
@config.register(nn.ConvTranspose3d)
@config.register(nn.Linear)
def handle_bias(mod):
    config = gather_params_from_instance(mod, exclude=('bias',))
    config['bias'] = mod.bias is not None
    return config


@config.register(nn.Embedding)
@config.register(nn.EmbeddingBag)
def handle_embedding(mod):
    return gather_params_from_instance(mod, exclude=('_weight',))


@config.register(nn.FractionalMaxPool2d)
@config.register(nn.FractionalMaxPool3d)
def handle_frac_max_pool(mod):
    return gather_params_from_instance(mod, exclude=('_random_samples',))


@config.register(nn.GRU)
@config.register(nn.LSTM)
@config.register(nn.RNN)
def handle_rnn(mod):
    params = inspect.signature(mod.__class__.__bases__[0]).parameters
    return gather_params_from_instance(mod, params, exclude=('mode',))


@config.register(nn.Identity)
def handle_identity(mod):
    return {}


@config.register(nn.Sequential)
def handle_sequential(mod):
    return {'*args': [extended_config(m) for m in mod]}
