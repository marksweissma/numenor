from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from functools import reduce, singledispatch
from itertools import chain
from numbers import Number
from typing import *

import pandas as pd
from attr import Factory, asdict, define, evolve, field
from sklearn import model_selection as ms

from numenor.utils import (as_factory, enforce_first_arg,
                           pop_with_default_factory)

GENERIC_SPLIT_FOLD_TYPE = Union[ms.KFold, ms.ShuffleSplit]
"""
get_keys manages the key type
get_key manages the data type
"""

from enum import Enum


class DataComponents(Enum):
    INDEX = "index"
    FEATURES = "features"
    TARGET = "target"
    PREDICTION = "prediction"
    METADATA = "metadata"


@enforce_first_arg
@singledispatch
def get_keys(keys, data):
    keys = [get_key(data, key) for key in keys] if data is not None else []
    return [key for key in keys if key]


@get_keys.register(str)
@get_keys.register(int)
@get_keys.register(float)
def get_keys_if_string(keys, data):
    keys = [get_key(data, keys)]
    return [key for key in keys if key]


@enforce_first_arg
@singledispatch
def get_key(data, key):
    return key if key in data else None


@get_key.register(pd.Series)
def get_key_if_series(data, key):
    return key if key == data.name else None


@enforce_first_arg
@singledispatch
def extract_keys(data):
    keys = list(data.keys())
    return keys if keys else []


@extract_keys.register(pd.Series)
def extract_key_series(data, default_factory=list):
    return [data.name] if data.name else []


@enforce_first_arg
@singledispatch
def enforce_list(obj):
    return [obj]


@enforce_list.register(list)
def enforce_list_noop(obj):
    return obj


@enforce_first_arg
@singledispatch
def chain_enforce_list(first, *rest):
    return list(chain(enforce_list(first), *(enforce_list(i) for i in rest)))


@chain_enforce_list.register(type(None))
def chain_enforce_list_str(first, *rest):
    return list(chain(*(enforce_list(i) for i in rest)))


def combine_keys(keys, X, y=None, **kwargs):
    x_keys = get_keys(keys, X)
    y_keys = get_keys(keys, y)
    return x_keys + y_keys


def _get_split_type(split_class_name):
    return getattr(ms, split_class_name)


@enforce_first_arg
@singledispatch
def split_factory(split_class, split_params=None):
    split_params = split_params if split_params else {}
    return split_class(**split_params)


@split_factory.register(str)
def split_factory_from_module_attribute(split_class, split_params=None):
    return split_factory(getattr(ms, split_class), split_params)


def default_split_config():
    config = {"split_class": "ShuffleSplit", "split_params": {"random_state": 22}}
    return config


@define
class Split:
    executor: Any = field(
        converter=as_factory(split_factory), default=Factory(default_split_config)
    )
    stratify_keys: Iterable = ()
    group_keys: Iterable = ()

    def get_index_location_generator(
        self, anchor_data: Iterable
    ) -> Generator[Iterable]:
        stratification = (
            anchor_data[get_keys(self.stratify_keys, anchor_data)]
            if self.stratify_keys
            else None
        )
        groups = (
            anchor_data[get_keys(self.group_keys, anchor_data)]
            if self.group_keys
            else None
        )
        index_location_generator = self.executor.split(
            anchor_data, stratification, groups
        )
        return index_location_generator


@enforce_first_arg
@singledispatch
def dataset_enforcer(data: Union[Dict, Data]) -> List[Data]:
    return [as_factory(Data)(data)]


@dataset_enforcer.register(list)
@dataset_enforcer.register(tuple)
def dataset_enforcer_iterable(datas):
    return [as_factory(Data)(data) for data in datas]


@enforce_first_arg
@singledispatch
def partition_datas(key, datas):
    head = None
    tail = []
    for index, data in enumerate(datas):
        if index == key:
            head = data
        else:
            tail.append(data)
    if not head:
        raise ColumnError(f"key: {key} does not exist in {datas}")
    return head, tail


@partition_datas.register(str)
def partition_datas_by_name(key, datas):
    head = None
    tail = []
    for data in datas:
        if data.name == key:
            head = data
        else:
            tail.append(data)
    if not head:
        raise ColumnError(f"key: {key} does not exist in {datas}")
    return head, tail


def pandas_concat(output):
    return pd.concat(output, axis=1)


@singledispatch
def prefix_expand(match_key, keys):
    return [key for key in keys if str(key).startswith(str(match_key))]


@prefix_expand.register(list)
@prefix_expand.register(tuple)
def prefix_expand_collection(match_key, keys):
    return [
        key
        for key in keys
        if any(str(key).startswith(str(single_key)) for single_key in match_key)
    ]


@singledispatch
def infer(keyset, keys):
    return keys


@infer.register(str)
@infer.register(Number)
def infer_str(keyset, keys):
    if len(keys) == 1:
        output = keys[0]
    else:
        output = keys
    return output


@define
class Column:
    column: Union[Hashable, List[Hashable]] = None
    matcher: Callable = prefix_expand
    postprocess: Callable = infer

    @classmethod
    def infer_from_shape(cls, arr, **kwargs):
        if hasattr(arr, "ndim") and arr.ndim > 1 and hasattr(arr, "columns"):
            column = list(arr.columns)
        elif hasattr(arr, "ndim") and arr.ndim == 1 and hasattr(arr, "name"):
            column = arr.name
        elif hasattr(arr, "ndim") and arr.ndim > 1:
            column = list(range(arr.ndim))

        elif hasattr(arr, "ndim") and arr.ndim == 1:
            column = 0
        elif arr is None:
            column = None
        else:
            ValueError(f"arr {arr} not a valid shape to infer from")
        return cls(column, **kwargs)  #  type: ignore

    def find_matches(self, matchable_columns: List[Hashable]):
        matches = self.matcher(self.column, matchable_columns)
        return self.postprocess(self.column, matches)

    def __bool__(self):
        return bool(self.column)


@define
class Data:
    table: Any
    feature_column: Column = field(
        converter=as_factory(Column), default=Factory(Column)
    )
    target_column: Column = field(converter=as_factory(Column), default=Factory(Column))
    prediction_column: Column = field(
        converter=as_factory(Column), default=Factory(Column)
    )
    index_column: Column = field(converter=as_factory(Column), default=Factory(Column))
    metadata_column: Column = field(
        converter=as_factory(Column), default=Factory(Column)
    )
    concat: Optional[Callable] = pandas_concat
    name: Optional[Union[str, int]] = None

    @classmethod
    def from_table(cls, table, **params):
        return cls(table, **params)

    @classmethod
    def from_components(
        cls,
        features=None,
        index=None,
        target=None,
        prediction=None,
        metadata=None,
        **params,
    ):

        feature_column = Column.infer_from_shape(features)
        index_column = Column.infer_from_shape(index)
        target_column = Column.infer_from_shape(target)
        prediction_column = Column.infer_from_shape(prediction)
        metadata_column = Column.infer_from_shape(metadata)

        _concat = params.get("concat", pandas_concat)
        block = [i for i in (features, target, prediction, metadata) if i is not None]
        table = _concat(block) if block else []

        params.update(
            {
                "feature_column": feature_column,
                "index_column": index_column,
                "target_column": target_column,
                "metadata_column": metadata_column,
                "prediction_column": prediction_column,
            }
        )

        return cls.from_table(table, **params)

    @property
    def columns(self):
        return list(self.table)

    @property
    def table_index(self):
        return self.table.index

    @property
    def index(self):
        columns = self.index_column.find_matches(self.columns)
        return self.table[columns] if columns else self.table_index

    @property
    def non_feature_column(self):
        return (
            self.index_column,
            self.metadata_column,
            self.prediction_column,
            self.target_column,
        )

    @property
    def feature_keys(self):
        if self.feature_column:
            columns = self.feature_column.find_matches(self.columns)
        else:
            non_feature_columns = [
                column.find_matches(self.columns)
                for column in self.non_feature_column
                if column
            ]
            exclude_columns = reduce(
                lambda x, y: x.union(enforce_list(y)) if y else x,
                [set([]), *non_feature_columns],
            )
            columns = [i for i in self.table if i not in exclude_columns]
        return columns

    @property
    def features(self):
        return self.table.get(self.feature_keys)

    @property
    def target(self):
        columns = self.target_column.find_matches(self.columns)
        return self.table.get(columns)

    @property
    def prediction(self):
        columns = self.prediction_column.find_matches(self.columns)
        return self.table.get(columns)

    @property
    def metadata(self):
        columns = self.metadata_column.find_matches(self.columns)
        return self.table.get(columns)

    def append_columns(self, data, location=None, **updates):
        keys = extract_keys(data)
        keys = keys if keys else [location]
        updates = deepcopy(updates)

        if location:
            location_key = f"{location}_key"
            updates.update(
                {location_key: chain_enforce_list(keys, getattr(self, location_key))}
            )

        table = self.concat([self.table, data]) if self.table is not None else data
        return evolve(self, table=table, **updates)

    def get_indices_from_index_location(self, ilocs):
        try:
            indices = self.index.iloc[ilocs]
        except AttributeError as e:
            if isinstance(self.index, pd.Index):
                indices = self.index[ilocs]
            else:
                raise e
        return indices

    def from_index_locations(self, ilocs, **updates):
        table = self.table.iloc[ilocs]
        return evolve(self, table=table, **updates)

    def from_index_values(self, index_values, **updates):
        if not self.index_column:
            table = self.table.loc[index_values]
        else:
            table = self.table.loc[self.index.isin(index_values)]
        return evolve(self, table=table, **updates)

    def groupby(self, group_key):
        return (
            (name, evolve(self, table=_df))
            for name, _df in self.table.groupby(group_key)
        )


@define
class Dataset:
    datas: List[Data] = field(converter=dataset_enforcer)
    splitter: Split = field(converter=as_factory(Split), default=Factory(Split))
    children: Sequence = Factory(list)
    anchor_data_key: Optional[int | str] = None

    @classmethod
    def from_datas(cls, datas, **kwargs):
        return cls(datas, **kwargs)

    @classmethod
    def from_method(
        cls,
        method="from_components",
        splitter: Optional[Split] = None,
        children: Optional[Sequence] = None,
        anchor_data_key: Optional[int | str] = None,
        **data_kwargs,
    ):
        splitter = splitter if splitter else Split()
        children = children if children else []
        return cls(
            getattr(Data, method)(**data_kwargs), splitter, children, anchor_data_key
        )

    def _build_children_split_confs(self):
        heads = self.children[0]
        if isinstance(heads, dict):
            heads = [heads]
        if len(heads) == 1:
            heads = [heads[0], deepcopy(heads[0])]

        [head.update({"children": None}) for head in heads if "children" not in head]
        [
            asdict(self, recurse=False, filter=lambda x, y: x.name != "datas").update(
                head
            )
            for head in heads
        ]
        return heads

    def build_children_params(self):
        if not self.children:
            params = asdict(self, recurse=False, filter=lambda x, y: x.name != "datas")
            confs = [deepcopy(params), deepcopy(params)]
        else:
            confs = self._build_children_split_confs()
        return confs

    @property
    def anchor(self):
        anchor, _ = self.get_anchor_rest_data()
        return anchor

    def get_anchor_rest_data(self):
        if self.anchor_data_key is None:
            anchor, rest = self.datas[0], self.datas[1:]
        else:
            anchor, rest = partition_datas(self.anchor_data_key, self.datas)
        return anchor, rest

    def get_index_location_generator(self, data):
        return self.splitter.get_index_location_generator(data.table)

    def split(self, index_location_generator=None, anchor=None):
        anchor, _ = self.get_anchor_rest_data() if anchor is None else (anchor, None)
        index_location_generator = (
            index_location_generator
            if index_location_generator is not None
            else self.get_index_location_generator(anchor)
        )

        train_conf, test_conf = self.build_children_params()
        train_index_locations, test_index_locations = next(index_location_generator)

        train_indices = anchor.get_indices_from_index_location(train_index_locations)
        test_indices = anchor.get_indices_from_index_location(test_index_locations)

        train_data = []
        test_data = []
        for data in self.datas:
            train_data.append(
                data.from_index_values(
                    train_indices, **pop_with_default_factory(train_conf, data.name)
                )
            )
            test_data.append(
                data.from_index_values(
                    test_indices, **pop_with_default_factory(test_conf, data.name)
                )
            )

        train_dataset = evolve(self, datas=train_data, **train_conf)
        test_dataset = evolve(self, datas=test_data, **test_conf)

        return train_dataset, test_dataset

    def split_iterator(self, index_location_generator=None, n_splits=5):
        anchor, _ = self.get_anchor_rest_data()
        index_location_generator = (
            index_location_generator
            if index_location_generator is not None
            else self.get_index_location_generator(anchor)
        )
        while n_splits:
            n_splits -= 1
            yield self.split(index_location_generator, anchor)

    def get_anchor_table_and_split_generator(self):
        anchor, rest = self.get_anchor_rest_data()
        generator = self.splitter.get_index_location_generator(anchor.table)
        return anchor, generator

    def reduce(self):
        anchor, rest = self.get_anchor_rest_data()
        return anchor

    @property
    def features(self):
        anchor = self.reduce()
        return anchor.features

    @property
    def target(self):
        anchor = self.reduce()
        return anchor.target

    @property
    def prediction(self):
        anchor = self.reduce()
        return anchor.prediction

    @property
    def metadata(self):
        anchor = self.reduce()
        return anchor.metadata

    def __len__(self):
        return len(self.reduce().table)

    def __getitem__(self, key):
        if key == 0:
            value = self.features
        elif key == 1:
            value = self.target
        else:
            raise KeyError(f"Key map only supported for (0, 1) not {key}")
        return value

    def __iter__(self):
        for _attribute in ["features", "target"]:
            yield getattr(self, _attribute)
