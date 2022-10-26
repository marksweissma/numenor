import pytest

from tutorials import test_script, without_config


@pytest.mark.example
@pytest.mark.parametrize(
    "example", [without_config.regression_example, test_script.test_end_to_end]
)
def test_examples_without_config(example):
    variants = example._variants
    [example(variant, plot=False) for variant in variants]
