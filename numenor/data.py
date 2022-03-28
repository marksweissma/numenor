import attr

from sklearn import model_selection as ms
from sklearn.base import BaseEstimator
import pandas as pd
from functools import singledispatch

from numenor.utils import as_factory, enforce_first_arg, pop_with_default_factory
from typing import *
from copy import deepcopy
'''
get_keys manages the key type
get_key manages the data type
'''

GENERIC_SPLIT_FOLD_TYPE = Union[ms.KFold, ms.ShuffleSplit]


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
def extract_key_seriess(data, default_factory=list):
    return [data.name] if data.name else []


@enforce_first_arg
@singledispatch
def append_key(chain, key):
    return chain + [key]


@append_key.register(str)
def append_key_str(chain, key):
    return [chain] + [key]


@append_key.register(type(None))
def append_key_str(chain, key):
    return [key]


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
    config = {
        'split_class': 'ShuffleSplit',
        'split_params': {
            'random_state': 22
        }
    }
    return config


@attr.s(auto_attribs=True)
class Split(BaseEstimator):
    executor: Any = attr.ib(converter=as_factory(split_factory),
                            default=attr.Factory(default_split_config))
    stratify_keys: Iterable = ()
    group_keys: Iterable = ()

    def get_params(self):
        return attr.asdict(self)

    def get_index_location_generator(self, anchor_data):
        stratification = anchor_data[get_keys(
            self.stratify_keys, anchor_data)] if self.stratify_keys else None
        groups = anchor_data[get_keys(
            self.group_keys, anchor_data)] if self.group_keys else None
        index_location_generator = self.executor.split(anchor_data,
                                                       stratification, groups)
        return index_location_generator


@enforce_first_arg
@singledispatch
def dataset_enforcer(data):
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
        raise KeyError(f'key: {key} does not exist in {datas}')
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
        raise KeyError(f'key: {key} does not exist in {datas}')
    return head, tail


def pandas_concat(output):
    return pd.concat(output, axis=1)


@attr.s(auto_attribs=True)
class Data(BaseEstimator):
    table: Any
    target_key: Optional[Iterable] = None
    prediction_key: Optional[Iterable] = None
    index_key: Optional[Iterable] = None
    concat: Optional[Callable] = pandas_concat
    name: Optional[Union[str, int]] = None

    @classmethod
    def from_table(cls, table, **params):
        return cls(table, **params)

    @classmethod
    def from_components(cls,
                        features=None,
                        target=None,
                        prediction=None,
                        **params):
        output = []
        target_key = None
        prediction_key = None

        output.append(features) if features is not None else None

        output.append(target) if target is not None else None
        target_key = get_key(target) if target is not None else None

        output.append(prediction) if prediction is not None else None
        prediction_key = get_key(
            prediction) if prediction is not None else None

        table = self.concat(output) if output else []

        params.update({'target_key': target_key}) if target_key else None
        params.update({'prediction_key': prediction_key
                       }) if prediction_key else None

        return cls.from_table(table, **params)

    @property
    def raw_index(self):
        return self.table.index

    @property
    def index(self):
        return self.table[
            self.index_key] if self.index_key else self.table.index

    @property
    def features(self):
        exclusions = append_key(self.target_key, self.prediction_key)
        columns = [i for i in self.table if i not in exclusions]
        return self.table.get(columns)

    @property
    def target(self):
        return self.table.get(self.target_key)

    @property
    def prediction(self):
        return self.table.get(self.prediction_key)

    def get_params(self, deep=True, exclusions=('table')):
        params = {}
        for key in filter(lambda x: x not in exclusions,
                          self._get_param_names()):
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                params[key] = value.get_params()
            else:
                params[key] = value
        return params

    def clone_with_params(self, method='from_table', **updates):
        params = self.get_params()
        params.update(updates)
        return getattr(self, method)(**params)

    def append_columns(self, data, location=None, **updates):
        keys = extract_keys(data)
        keys = keys if keys else [location]
        params = self.get_params()
        params.update(updates)
        location_key = f'{location}_key'
        params.update(
            {location_key: append_key(keys, getattr(self, location_key))})
        table = self.concat([self.table, data
                             ]) if self.table is not None else data
        return self.from_table(table, **params)

    def get_indices_from_index_location(self, ilocs):
        indices = self[
            self.
            index_key].iloc[ilocs] if self.index_key else self.index[ilocs]
        return indices

    def from_index_locations(self, ilocs, **updates):
        table = self.iloc[ilocs]
        return self.clone_with_params(table=table, **updates)

    def from_index_locations(self, locs, **updates):
        table = self.loc[locs]
        return self.clone_with_params(table=table, **updates)

    def from_index_values(self, index_values):
        indices = self[self.index_key].isin(
            index_values) if self.index_key else index_values
        return self.from_index_locations(index_mask)

    def __getattr__(self, key):
        return getattr(self.table, key)


@attr.s(auto_attribs=True)
class Dataset(BaseEstimator):
    datas: List[Data] = attr.ib(converter=dataset_enforcer)
    splitter: Split = attr.ib(converter=as_factory(Split),
                              default=attr.Factory(Split))
    children: Sequence = attr.Factory(list)
    anchor_data_key: Union[int, str] = None

    @classmethod
    def from_datas(cls, datas, **kwargs):
        return cls(datas, **kwargs)

    def _build_children_split_confs(self):
        heads = self.children[0]
        if isinstance(heads, dict):
            heads = [heads]
        if len(heads) == 1:
            heads = [heads[0], deepcopy(heads[0])]

        [
            head.update({'children': None}) for head in heads
            if 'children' not in head
        ]
        [self.get_params(deep=False).update(head) for head in heads]
        return heads

    def build_children_params(self):
        if not self.children:
            params = self.get_params(deep=False)
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
        anchor, _ = self.get_anchor_rest_data() if anchor is None else (anchor,
                                                                        None)
        index_location_generator = index_location_generator if index_location_generator is not None else self.get_index_location_generator(
            anchor)

        train_conf, test_conf = self.build_children_params()
        train_index_locations, test_index_locations = next(
            index_location_generator)

        train_indices = anchor.get_indices_from_index_location(
            train_index_locations)
        test_indices = anchor.get_indices_from_index_location(
            test_index_locations)

        train_data = []
        test_data = []
        for data in self.datas:
            train_data.append(
                data.from_index_locations(
                    train_indices,
                    **pop_with_default_factory(train_conf, data.name)))
            test_data.append(
                data.from_index_locations(
                    test_indices,
                    **pop_with_default_factory(test_conf, data.name)))

        train_dataset = self.clone_with_params(train_data, **train_conf)
        test_dataset = self.clone_with_params(test_data, **test_conf)

        return train_dataset, test_dataset

    def split_iterator(self, index_location_generator=None, n_splits=5):
        anchor, _ = self.get_anchor_rest_data()
        index_location_generator = index_location_generator if index_location_generator is not None else self.get_index_location_generator(
            anchor)
        while n_splits:
            n_splits -= 1
            yield self.split(index_location_generator, anchor)

    def clone_with_params(self, datas, method='from_datas', **updates):
        params = self.get_params(deep=False)
        params.update(updates)
        return getattr(self, method)(datas, **params)

    def get_params(self, deep=True, exclusions=('datas')):
        params = {}
        for key in filter(lambda x: x not in exclusions,
                          self._get_param_names()):
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                params[key] = value.get_params()
            else:
                params[key] = value
        return params

    def get_anchor_table_and_split_generator(self):
        anchor, rest = self.get_anchor_rest_data()
        generator = self.splitter.get_index_location_generator(anchor.table)
        return anchor, generator
