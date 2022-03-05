from functools import lru_cache, singledispatch, reduce
from wrapt import decorator
from inspect import signature


@lru_cache(1000)
def get_signature_params(func):
    return signature(func).parameters


@decorator
def enforce_first_arg(wrapped, instance, args, kwargs):
    if not args:
        parameters = get_signature_params(wrapped)
        args = (kwargs.pop(list(parameters)[0]),)
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


singledispatch = enforce_first_arg(singledispatch)
