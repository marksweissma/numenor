import pytest

import pandas as pd
from numenor import data
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit

SPLITTER = data.Split(executor=ShuffleSplit(test_size=.4))
STRATIFIED_SPLITTER = data.Split(executor=StratifiedShuffleSplit(test_size=.5),
                                 stratify_keys='color')
GROUP_SPLITTER = data.Split(executor=GroupShuffleSplit(test_size=.5),
                            group_keys=['color'])
OTHER_STRATIFIED_SPLITTER = data.Split(
    executor=StratifiedShuffleSplit(test_size=.5), stratify_keys='tranche')


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


def test_get_keys_df(df):
    assert data.get_keys(['a'], df) == ['a']
    assert data.get_keys('a', df) == ['a']
    assert data.get_keys(1, df) == [1]
    assert data.get_keys(2.0, df) == [pytest.approx(2.0)]
    assert data.get_keys('d', df) == []
    assert data.get_keys(['d'], df) == []
    assert data.get_keys([1, 'a', 'd'], df) == [1, 'a']
    assert data.get_keys([], df) == []


def test_get_keys_series(series):
    assert data.get_keys(['a'], series) == ['a']
    assert data.get_keys('a', series) == ['a']
    assert data.get_keys(1, series) == []


def test_dataset_split(df):
    dataset = data.Dataset([data.Data(df)])
    train, test = dataset.split()
    m = len(train.datas[0].table)
    n = len(test.datas[0].table)
    assert n / (m + n) == pytest.approx(.1)

    # datas conversion
    dataset = data.Dataset(data.Data(df))
    train, test = dataset.split()
    m = len(train.datas[0].table)
    n = len(test.datas[0].table)
    assert n / (m + n) == pytest.approx(.1)

    test_size = 1 / 9
    train.splitter.test_size = test_size
    train_train, train_test = train.split()
    m = len(train_train.datas[0].table)
    n = len(train_test.datas[0].table)
    assert n / (m + n) == pytest.approx(test_size)


def test_children(df):
    children = [{'splitter': SPLITTER}]
    ds = data.Dataset([data.Data(df)],
                      splitter=GROUP_SPLITTER,
                      children=children)
    train, test = ds.split()

    assert train.datas[0].table['color'].nunique() == 1
    assert test.datas[0].table['color'].nunique() == 1
    assert len(train.datas[0].table) == len(test.datas[0].table)

    train_a, train_b = train.split()
    assert len(train_a.datas[0].table) > len(train_b.datas[0].table)
    assert len(train_b.datas[0].table) > 0


def test_multiple_children(df):
    children = [[{
        'splitter':
        data.Split(executor=GroupShuffleSplit(test_size=.4),
                   group_keys=['tranche'])
    }, {
        'splitter':
        data.Split(executor=StratifiedShuffleSplit(test_size=.5),
                   stratify_keys=['cut'])
    }]]

    ds = data.Dataset([data.Data(df)],
                      splitter=GROUP_SPLITTER,
                      children=children)
    train, test = ds.split()
    train_a, train_b = train.split()
    assert len(
        set(train_a.datas[0].table.tranche).intersection(
            train_b.datas[0].table.tranche)) == 0

    test_a, test_b = test.split()
    assert set(test_a.datas[0].table.tranche) == set(
        test_b.datas[0].table.tranche)
