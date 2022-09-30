from __future__ import annotations

from functools import singledispatch

import pytest
from attrs import define
from wrapt import decorator

from numenor.utils import (as_factory, call_from_attribute_or_callable,
                           enforce_first_arg, get_attribute_or_call,
                           pop_with_default_factory)


def test_enforce_first_arg():
    @enforce_first_arg
    def func(a, **kwargs):
        return a

    payload = {"b": 1, "a": 2}
    assert func(**payload) == payload["a"]

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

    assert fixed_func(**payload) == payload["a"]

    @fixed_func.register(int)
    def fixed_func_str(a, **kwargs):
        return -a

    assert fixed_func(**payload) == -payload["a"]


@pytest.fixture
def example():
    @define
    class A:
        b: int = 1

        @staticmethod
        def calculate(a: A, value: int = 1):
            return a.b + value

        def augment(self, value: int = 1):

            return self.b + value

    return A()


def test_get_attribute_or_call(example):
    assert get_attribute_or_call("b", example) == 1
    assert get_attribute_or_call(example.calculate, example) == 2
    assert get_attribute_or_call(example.calculate, example, 2) == 3
    assert get_attribute_or_call(example.calculate, example, value=3) == 4


def test_call_from_attribute_or_callable(example):
    assert call_from_attribute_or_callable("augment", example, 0) == 1
    assert call_from_attribute_or_callable(example.calculate, example, 2) == 3
    assert call_from_attribute_or_callable(example.calculate, example, value=3) == 4
