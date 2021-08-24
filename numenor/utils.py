from functools import singledispatch


def as_factory(handler):
    """
    converter for handler to accept object or a `dict` config
    for a handler to generate an object. handlers are callable
    (functions or types). Conceptually the hope is it feels like
    and hopefully works well with google's fire
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
