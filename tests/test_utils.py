import pytest
from functools import singledispatch
from numenor.utils import enforce_first_arg, as_factory, append_key, compose_decorators
from wrapt import decorator


def test_compose_decorators():

    def annotate(value):

        @decorator
        def wrapper(wrapped, instance, args, kwargs):
            return [value] + wrapped(*args, **kwargs) + [value]

        return wrapper

    @compose_decorators(annotate('first'), annotate('second'))
    def func(value='func'):
        return [value]

    assert func() == ['first', 'second', 'func', 'second', 'first']


def test_enforce_first_arg():

    @enforce_first_arg
    def func(a, **kwargs):
        return a

    payload = {'b': 1, 'a': 2}
    assert func(**payload) == payload['a']

    @singledispatch
    def broken_func(a, **kwargs):
        return a

    with pytest.raises(TypeError):
        broken_func(**payload)

    @broken_func.register(int)
    def broken_func_str(a, **kwargs):
        return -a

    with pytest.raises(TypeError):
        broken_func(**payload)

    @enforce_first_arg
    @singledispatch
    def fixed_func(a, **kwargs):
        return a

    assert fixed_func(**payload) == payload['a']

    @fixed_func.register(int)
    def fixed_func_str(a, **kwargs):
        return -a

    assert fixed_func(**payload) == -payload['a']
