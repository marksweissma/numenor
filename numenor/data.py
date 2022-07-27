from attr import define, field, Factory, evolve, asdict
from itertools import chain

from sklearn import model_selection as ms
import pandas as pd
from functools import singledispatch, reduce

from numenor.utils import as_factory, enforce_first_arg, pop_with_default_factory
from typing import *
from copy import deepcopy
from collections.abc import Iterable

GENERIC_SPLIT_FOLD_TYPE = Union[ms.KFold, ms.ShuffleSplit]
'''
get_keys manages the key type
get_key manages the data type
'''


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
def chain_enforce_list(chain, key):
    return chain + [key]


@chain_enforce_list.register(str)
def chain_enforce_list_str(chain, key):
    return [chain] + [key]


@chain_enforce_list.register(type(None))
def chain_enforce_list_str(chain, key):
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


@define
class Split:
    executor: Any = field(converter=as_factory(split_factory),
                          default=Factory(default_split_config))
    stratify_keys: Iterable = ()
    group_keys: Iterable = ()

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


def prefix_expand(key, table):
    return [column for column in table if str(column).startswith(str(key))]


@define
class Data:
    table: Any
    target_keys: Optional[Iterable] = None
    prediction_keys: Optional[Iterable] = None
    index_key: Optional[Iterable] = None
    metadata_keys: Optional[Iterable] = None
    concat: Optional[Callable] = pandas_concat
    name: Optional[Union[str, int]] = None
    expand: Callable = prefix_expand

    @classmethod
    def from_table(cls, table, **params):
        return cls(table, **params)

    @classmethod
    def from_components(cls,
                        features=None,
                        target=None,
                        prediction=None,
                        metadata=None,
                        **params):
        output = []
        target_keys = None
        metadata_keys = None
        prediction_keys = None

        def append_output_and_get(block):
            output.append(block) if block is not None else None
            return get_key(block) if block is not None else None

        _ = append_output_and_get_block(features)

        target_keys = append_output_and_get_block(target)
        metadata_keys = append_output_and_get_block(metadata)
        prediction_keys = append_output_and_get_block(prediction)

        _concat = params.get('concat', pandas_concat)
        table = _concat(output) if output else []

        params.update({'target_keys': target_keys}) if target_keys else None
        params.update({'metadata_keys': metadata_keys
                       }) if metadata_keys else None
        params.update({'prediction_keys': prediction_keys
                       }) if prediction_keys else None

        return cls.from_table(table, **params)

    @property
    def table_index(self):
        return self.table.index

    @property
    def index(self):
        return self.table[
            self.index_key] if self.index_key else self.table_index

    @property
    def non_feature_keysets(self):
        keysets = reduce(lambda x, y: chain_enforce_list(x, y), [None] + [
            self.target_keys, self.prediction_keys, self.metadata_keys,
            self.index_key
        ])
        print(keysets)
        return [keyset for keyset in keysets if keyset]

    @property
    def feature_keys(self):
        exclusions = list(
            chain.from_iterable(
                self.expand(keys, self.table)
                for keys in self.non_feature_keysets))
        return [column for column in self.table if column not in exclusions]

    @property
    def features(self):
        return self.table.get(self.feature_keys)

    @property
    def target(self):
        return self.table.get(self.target_keys)

    @property
    def metadata(self):
        return self.table.get(self.metadata_keys)

    @property
    def prediction(self):
        return self.table.get(self.prediction_keys)

    def append_columns(self, data, location=None, **updates):
        keys = extract_keys(data)
        keys = keys if keys else [location]
        updates = deepcopy(updates)

        if location:
            location_key = f'{location}_key'
            updates.update({
                location_key:
                chain_enforce_list(keys, getattr(self, location_key))
            })

        table = self.concat([self.table, data
                             ]) if self.table is not None else data
        return evolve(self, table=table, **updates)

    def get_indices_from_index_location(self, ilocs):
        indices = self[
            self.
            index_key].iloc[ilocs] if self.index_key else self.index[ilocs]
        return indices

    def from_index_locations(self, ilocs, **updates):
        table = self.iloc[ilocs]
        return evolve(self, table=table, **updates)

    def from_index_locations(self, locs, **updates):
        table = self.loc[locs]
        return evolve(self, table=table, **updates)

    def from_index_values(self, index_values):
        indices = self[self.index_key].isin(
            index_values) if self.index_key else index_values
        return self.from_index_locations(index_mask)

    def groupby(self, group_key):
        return ((name, evolve(self, table=_df))
                for name, _df in self.table.groupby(group_key))

    def __getattr__(self, key):
        return getattr(self.table, key)


@define
class Dataset:
    datas: List[Data] = field(converter=dataset_enforcer)
    splitter: Split = field(converter=as_factory(Split),
                            default=Factory(Split))
    children: Sequence = Factory(list)
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
        [
            asdict(self, recurse=False,
                   filter=lambda x, y: x.name != 'datas').update(head)
            for head in heads
        ]
        return heads

    def build_children_params(self):
        if not self.children:
            params = asdict(self,
                            recurse=False,
                            filter=lambda x, y: x.name != 'datas')
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

        train_dataset = evolve(self, datas=train_data, **train_conf)
        test_dataset = evolve(self, datas=test_data, **test_conf)

        return train_dataset, test_dataset

    def split_iterator(self, index_location_generator=None, n_splits=5):
        anchor, _ = self.get_anchor_rest_data()
        index_location_generator = index_location_generator if index_location_generator is not None else self.get_index_location_generator(
            anchor)
        while n_splits:
            n_splits -= 1
            yield self.split(index_location_generator, anchor)

    def get_anchor_table_and_split_generator(self):
        anchor, rest = self.get_anchor_rest_data()
        generator = self.splitter.get_index_location_generator(anchor.table)
        return anchor, generator
