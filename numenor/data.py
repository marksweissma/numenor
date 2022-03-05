import attr

from sklearn import model_selection as ms
from sklearn.base import BaseEstimator
import pandas as pd

from numenor.utils import as_factory, enforce_first_arg, singledispatch
from typing import *

'''
get_keys manages the key type
get_key manages the data type
'''


GENERIC_SPLIT_FOLD_TYPE = Union[ms.KFold, ms.ShuffleSplit]


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


@singledispatch
def get_key(data, key):
    return key if key in data else None

@get_keys.register(pd.Series)
def get_key_if_series(data, key):
    return key if key == data.name else None


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
    config = {'split_class': 'ShuffleSplit', 'split_params': {'random_state': 22}}
    return config


@attr.s(auto_attribs=True)
class Split(BaseEstimator):
    executor: Any = attr.ib(converter=as_factory(split_factory),
                            default=attr.Factory(default_split_config)
                            )
    stratify_keys: Iterable = ()
    group_keys: Iterable = ()

    def get_params(self):
        return attr.asdict(self)

    def get_index_location_generator(self, *args):
        stratification = combine_keys(self.stratify_keys, *args) if self.stratify_keys else None
        groups = combine_keys(self.group_keys, *args) if self.group_keys else None
        index_location_generator = self.executor.split(args[0], stratification, groups)
        return index_location_generator


@singledispatch
def dataset_enforcer(data):
    return [as_factory(Data)(data)]


@dataset_enforcer.register(list)
@dataset_enforcer.register(tuple)
def dataset_enforcer_iterable(datas):
    return [as_factory(Data)(data) for data in datas]


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
    target_key: Iterable = None
    prediction_key: Iterable = None
    index_key: Iterable = None
    concat: Callable = pandas_concat
    name: Union[str, int] = None

    @classmethod
    def from_table(cls, table, **params):
        return cls(table, **params)

    @classmethod
    def from_components(cls, features=None, target=None, prediction=None, **params):
        output = []
        target_key = None
        prediction_key = None
        if features is not None:
            output.append(features)
        if target is not None:
            output.append(target)
            target_key = get_key(target)
        if prediction is not None:
            output.append(prediction)
            prediction_key = get_key(prediction)

        if output:
            table = self.concat(output)

        if target_key:
            params['target_key'] = target_key
        if prediction_key:
            params['prediction_key'] = target_key

        return cls.from_table(table, **params)

    @property
    def index(self):
        return self.table[self.index_key] if self.index_key else self.table.index

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
        for key in filter(lambda x: x not in exclusions, self._get_param_names()):
            value = getattr(self, key)
            if deep and hasaattr(value, 'get_params'):
                params[key] = value.get_params()
            else:
                params[key] = value
        return params

    def clone_with_params(self, method='from_table', **updates):
        params = self.get_params()
        params.update(updates)
        return getattr(self, method)(**params)

    def append_columns(self, data, location, **updates):
        key = get_key(data)
        location_key = f'{location}_key'
        params = self.get_params()
        params.update(updates)
        params[location_key] = append_key(key, getattr(self, location_key))
        table = self.concat([self.table, data]) if self.table is not None else data
        return self.from_table(table, **params)

    def from_index_locations(self, ilocs, **updates):
        table = self.iloc[ilocs]
        return self.with_params(table=table, **updates)

    def from_index_locations(self, locs, **updates):
        table = self.loc[locs]
        return self.with_params(table=table, **updates)

    def from_index_values(self, index_values):
        indexer = self[self.indexing_key] if self.indexing_key is not None else self.index
        index_mask = indexer.isin(index_values)
        return self.from_index_locations(index_mask)

    def __getattr__(self, key):
        return getattr(self.table, key)


@attr.s(auto_attribs=True)
class Dataset(BaseEstimator):
    datas: List[Data] = attr.ib(converter=dataset_enforcer)
    splitter: Split = attr.ib(converter=as_factory(Split), default=attr.Factory(Split))
    children: Sequence = attr.Factory(list)

    anchor_data_key: Union[int, str] = None


    def _build_children_split_confs(self):
        heads = self.children[0]
        if isnstance(heads, dict):
            heads = [heads]
        if len(heads) == 1:
            heads = replicate(heads[0])

        [head.update({'children': None}) for head in heads if 'children' not in head]
        [self.get_params().update(head) for head in heads]
        return heads

    def build_children_params(self):
        if not self.children:
            confs = replicate(self.get_params())
        else:
            confs = self._build_children_split_confs()
        return confs

    def split(self):
        if self.anchor_data_key is None and len(self.datas) == 1:
            anchor = self.datas[0]
            rest = []
        else:
            anchor, rest = partition_datas(self.anchor_data_key, self.datas)
        generator = self.splitter.get_index_location_generator(anchor.table)
        train_index_locations, test_index_locations = next(generator)

        train_indices = anchor.get_indices_from_index_location(train_index_locations)
        test_indices = anchor.get_indices_from_index_location(test_index_locations)
        train_conf, test_conf = self.build_children_params()

        train_data = []
        test_data = []
        for data in self.datas:
            train_data.append(data.from_index_location(**train_conf))
            test_data.append(data.from_index_location(**test_conf))

        train_dataset = self.with_params(train_data)
        test_dataset = self.with_params(test_data)

        return train_dataset, test_dataset
