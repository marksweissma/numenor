from functools import lru_cache, reduce
from functools import singledispatch
from wrapt import decorator
from inspect import signature
import inspect


def coalesce(*args):
    """
    SQL like coelesce, returns first non Falsey arg if type supports truthy / falses, otherwise compares to None
    """
    for arg in args:
        try:
            if arg:
                return arg
        except ValueError:
            if arg is not None:
                return arg
    return args[-1]


def get_signature_params(func):
    return signature(func).parameters


@decorator
def enforce_first_arg(wrapped, instance, args, kwargs):
    if not args:
        parameters = get_signature_params(wrapped)
        first_arg_key = list(parameters)[0]
        args = (kwargs[first_arg_key], )
        kwargs = {
            key: value
            for key, value in kwargs.items() if key != first_arg_key
        }

    return wrapped(*args, **kwargs)


def as_factory(handler):
    """
    converter for handler to accept object or a `dict` config
    for a handler to generate an object. handlers are callable
    (functions or types).
    """

    @singledispatch
    def _as(obj):
        return obj

    @_as.register(dict)
    def _as_from_conf(conf):
        obj = handler(**conf)
        return obj

    return _as


@singledispatch
def append_key(key, chain):
    if chain:
        if isinstance(chain, str):
            keychain = [chain, key]
        elif isinstance(chain, list):
            keychain = chain + [key]
        else:
            raise TypeError('chain needs to be list or string')
    else:
        keychain = key
    return keychain


@append_key.register(list)
def append_key_list(key, chain):
    if chain:
        if isinstance(chain, str):
            keychain = [chain] + key
        elif isinstance(chain, list):
            keychain = chain + key
        else:
            raise TypeError('chain needs to be list or string')
    else:
        keychain = key
    return keychain


def pop_with_default_factory(store, key_sequence, factory=dict):
    if not key_sequence:
        return factory()
    value = store if hasattr(store, 'pop') else factory()
    for key in key_sequence:
        value = value.pop(key) if hasattr(value, 'pop') else factory()
    return value


def find_value(func, args, kwargs, access_key, how='name'):
    """
    Find a value from a function signature
    """

    parameters = get_signature_params(func)
    keys = list(parameters.keys())

    try:
        index = keys.index(access_key) if how == 'name' else access_key
        defaults = [
            i.default for i in parameters.values()
            if i.default != inspect._empty
        ]
        offset = len(keys) - len(coalesce(defaults, []))
        default = defaults[index - offset] if index >= offset else None
        value = kwargs.get(access_key,
                           default) if index >= len(args) else args[index]
    except ValueError:
        value = kwargs.get(access_key, None)
    return value


def replace_value(func, args, kwargs, access_key, access_value):
    """
    Replace a value from a function signature
    """
    parameters = get_signature_params(func)
    keys = list(parameters.keys())
    index = keys.index(access_key)

    if index >= len(args):
        kwargs[access_key] = access_value
    else:
        args = list(args)
        args[index] = access_value
        args = tuple(args)
    return args, kwargs
