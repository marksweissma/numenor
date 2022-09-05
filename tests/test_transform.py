import pytest
import pandas as pd

from numenor import transform as t


@pytest.fixture
def series():
    return pd.Series(list(range(10)), name='a')


@pytest.fixture
def _df():
    return pd.DataFrame({
        'a': list(range(20)),
        'b': list(range(20, 40)),
        'c': list(range(40, 60)),
        1: list(range(60, 80)),
        2.0: list(range(80, 100)),
    })


@pytest.fixture
def df_prefixes(_df):
    str_column = [i for i in _df if isinstance(i, str)]
    return pd.concat([_df, _df[str_column].add_prefix('pre_')], axis=1)


@pytest.fixture
def color(_df):
    return _df.apply(lambda row: 'blue'
                     if row.name % 2 else 'red', axis=1).rename('color')


@pytest.fixture
def block(_df):
    return _df.apply(lambda row: pd.Series({
        'tranche': row.name % 3,
        'cut': row.name % 5
    }),
                     axis=1)


@pytest.fixture
def df(_df, color, block):
    return pd.concat([_df, color, block], axis=1)


def test_select(df):
    selector = t.Select(include={'by_key': {'include': ['a']}})
    selector(df).equals(df[['a']])
    assert isinstance(selector(df), pd.DataFrame)

    selector = t.Select(include={'by_key': {'exclusions': ['a']}})
    selector(df).equals(df.drop('a', axis=1))
    assert isinstance(selector(df), pd.DataFrame)

    selector = t.Select(include={'by_dtype': {'include': ['object']}})
    assert selector(df).equals(df[['color']])

    selector = t.Select(include={'by_dtype': {'include': ['number']}})
    assert selector(df).equals(df.drop('color', axis=1))
