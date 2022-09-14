from itertools import chain

import pandas as pd
import pytest
from sklearn.model_selection import (GroupShuffleSplit, ShuffleSplit,
                                     StratifiedShuffleSplit)

from numenor import data

RANDOM_STATE = 42
SPLITTER = data.Split(executor=ShuffleSplit(test_size=0.4))
STRATIFIED_SPLITTER = data.Split(
    executor=StratifiedShuffleSplit(test_size=0.5, random_state=RANDOM_STATE),
    stratify_keys="color",
)
GROUP_SPLITTER = data.Split(
    executor=GroupShuffleSplit(test_size=0.5, random_state=RANDOM_STATE),
    group_keys=["color"],
)
OTHER_STRATIFIED_SPLITTER = data.Split(
    executor=StratifiedShuffleSplit(test_size=0.5), stratify_keys="tranche"
)


@pytest.fixture
def series():
    return pd.Series(list(range(10)), name="a")


@pytest.fixture
def _df():
    return pd.DataFrame(
        {
            "a": list(range(20)),
            "b": list(range(20, 40)),
            "c": list(range(40, 60)),
            1: list(range(60, 80)),
            2.0: list(range(80, 100)),
        }
    )


@pytest.fixture
def df_prefixes(_df):
    str_column = [i for i in _df if isinstance(i, str)]
    return pd.concat([_df, _df[str_column].add_prefix("pre_")], axis=1)


@pytest.fixture
def color(_df):
    return _df.apply(lambda row: "blue" if row.name % 2 else "red", axis=1).rename(
        "color"
    )


@pytest.fixture
def block(_df):
    return _df.apply(
        lambda row: pd.Series({"tranche": row.name % 3, "cut": row.name % 5}), axis=1
    )


@pytest.fixture
def df(_df, color, block):
    return pd.concat([_df, color, block], axis=1)


def test_get_keys_df(df):
    assert data.get_keys(["a"], df) == ["a"]
    assert data.get_keys("a", df) == ["a"]
    assert data.get_keys(1, df) == [1]
    assert data.get_keys(2.0, df) == [pytest.approx(2.0)]
    assert data.get_keys("d", df) == []
    assert data.get_keys(["d"], df) == []
    assert data.get_keys([1, "a", "d"], df) == [1, "a"]
    assert data.get_keys([], df) == []


def test_get_keys_series(series):
    assert data.get_keys(["a"], series) == ["a"]
    assert data.get_keys("a", series) == ["a"]
    assert data.get_keys(1, series) == []


def test_dataset_split(df):
    dataset = data.Dataset([data.Data(df)])
    train, test = dataset.split()
    m = len(train.datas[0].table)
    n = len(test.datas[0].table)
    assert n / (m + n) == pytest.approx(0.1)

    # datas conversion
    dataset = data.Dataset(data.Data(df))
    train, test = dataset.split()
    m = len(train.datas[0].table)
    n = len(test.datas[0].table)
    assert n / (m + n) == pytest.approx(0.1)

    test_size = 1 / 9
    train.splitter.executor.test_size = test_size
    train_train, train_test = train.split()
    m = len(train_train.datas[0].table)
    n = len(train_test.datas[0].table)
    assert n / (m + n) == pytest.approx(test_size)


def test_children(df):
    children = [{"splitter": SPLITTER}]
    ds = data.Dataset([data.Data(df)], splitter=GROUP_SPLITTER, children=children)
    train, test = ds.split()

    assert train.datas[0].table["color"].nunique() == 1
    assert test.datas[0].table["color"].nunique() == 1
    assert len(train.datas[0].table) == len(test.datas[0].table)

    train_a, train_b = train.split()
    assert len(train_a.datas[0].table) > len(train_b.datas[0].table)
    assert len(train_b.datas[0].table) > 0


def test_multiple_children(df):
    children = [
        [
            {
                "splitter": data.Split(
                    executor=GroupShuffleSplit(
                        test_size=0.4, random_state=RANDOM_STATE
                    ),
                    group_keys=["tranche"],
                )
            },
            {
                "splitter": data.Split(
                    executor=StratifiedShuffleSplit(
                        test_size=0.5, random_state=RANDOM_STATE
                    ),
                    stratify_keys=["cut"],
                )
            },
        ]
    ]

    ds = data.Dataset([data.Data(df)], splitter=GROUP_SPLITTER, children=children)
    train, test = ds.split()
    train_a, train_b = train.split()
    assert (
        len(
            set(train_a.datas[0].table.tranche).intersection(
                train_b.datas[0].table.tranche
            )
        )
        == 0
    )

    test_a, test_b = test.split()
    assert set(test_a.datas[0].table.tranche) == set(test_b.datas[0].table.tranche)


def test_column(_df, df_prefixes):
    a_data = data.Data(_df)
    column = data.Column(["a"])
    matches = column.find_matches(a_data.columns)
    assert matches == ["a"]

    column = data.Column("a")
    matches = column.find_matches(a_data.columns)
    assert matches == "a"

    a_data = data.Data(df_prefixes)
    column = data.Column("pre_")
    matches = column.find_matches(a_data.columns)
    assert matches == ["pre_a", "pre_b", "pre_c"]

    a_data = data.Data(df_prefixes)
    column = data.Column(["pre_"])
    matches = column.find_matches(a_data.columns)
    assert matches == ["pre_a", "pre_b", "pre_c"]

    a_data = data.Data(_df)
    column = data.Column([2])
    matches = column.find_matches(a_data.columns)
    assert matches == [2]


def test_features(_df):
    a_data = data.Data(_df)

    assert a_data.features.equals(_df)

    a_data = data.Data(_df, index_column="a")
    assert a_data.features.equals(_df.drop("a", axis=1))

    a_data = data.Data(_df, metadata_column=["a", "b"])
    assert a_data.features.equals(_df.drop(["a", "b"], axis=1))

    a_data = data.Data(_df, index_column=1, metadata_column=["a", "b"])
    assert a_data.features.equals(_df.drop([1, "a", "b"], axis=1))


def test_data_base(_df):
    a_data = data.Data(_df)
    assert a_data.feature_keys == list(_df)
    assert a_data.index.equals(_df.index)
    assert a_data.table.equals(_df)


def test_data_index_column(_df):
    index_column = 2.0
    a_data = data.Data(_df, index_column=index_column)

    assert a_data.index.name == pytest.approx(index_column)

    assert a_data.feature_keys == [i for i in _df if i != index_column]
    assert a_data.index.equals(_df[index_column])

    assert a_data.table.equals(_df)


def test_data_target_column(_df):
    target_column = 1
    a_data = data.Data(_df, target_column=target_column)

    assert a_data.feature_keys == [i for i in _df if i != target_column]
    assert a_data.target.equals(_df[target_column])

    assert a_data.table.equals(_df)

    target_column = [1]
    a_data = data.Data(_df, target_column=target_column)

    assert a_data.feature_keys == [i for i in _df if i not in target_column]
    assert a_data.target.equals(_df[target_column])

    assert a_data.table.equals(_df)


def test_data_prediction_column(_df):
    prediction_column = 1
    a_data = data.Data(_df, prediction_column=prediction_column)

    assert a_data.feature_keys == [i for i in _df if i != prediction_column]
    assert a_data.prediction.equals(_df[prediction_column])

    assert a_data.table.equals(_df)


def test_data_metadata_column(_df):
    metadata_column = 1
    a_data = data.Data(_df, metadata_column=metadata_column)

    assert a_data.feature_keys == [i for i in _df if i != metadata_column]
    assert a_data.metadata.equals(_df[metadata_column])

    assert a_data.table.equals(_df)


def test_data_target_and_prediction_column(_df):
    target_column = 1
    prediction_column = "a"
    a_data = data.Data(
        _df, target_column=target_column, prediction_column=prediction_column
    )

    assert a_data.feature_keys == [
        i for i in _df if i not in (target_column, prediction_column)
    ]
    assert a_data.target.equals(_df[target_column])
    assert a_data.prediction.equals(_df[prediction_column])

    assert a_data.table.equals(_df)

    target_column = [1]
    prediction_column = ["a"]
    a_data = data.Data(
        _df, target_column=target_column, prediction_column=prediction_column
    )

    assert a_data.feature_keys == [
        i for i in _df if i not in target_column + prediction_column
    ]
    assert a_data.target.equals(_df[target_column])
    assert a_data.prediction.equals(_df[prediction_column])

    assert a_data.table.equals(_df)


def test_prefixes(df_prefixes):
    metadata_prefix = "pre"
    metadata_column = [i for i in df_prefixes if str(i).startswith("pre")]
    a_data = data.Data(df_prefixes, metadata_column=metadata_prefix)

    assert a_data.feature_keys == [i for i in df_prefixes if i not in metadata_column]
    assert a_data.metadata.equals(df_prefixes[metadata_column])

    assert a_data.table.equals(df_prefixes)
