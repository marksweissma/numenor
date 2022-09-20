import pytest

from numenor_examples import without_config


@pytest.mark.example
@pytest.mark.parametrize("example", [without_config.regression_example])
def test_examples_without_config(example):
    variants = example._variants
    [example(variant, plot=False) for variant in variants]
