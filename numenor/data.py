import attr
from functools import singledispatch
from .utils import as_factory

"""
get_keys manages the key type
get_key manages the data type
"""

@singledispatch
def get_keys(keys, data):
    keys = [get_key(data, key) for key in keys]
    return [key for key in keys if key]

@get_keys.register(str)
@get_keys.register(int)
@get_keys.register(float)
def get_key_if_string(keys, data):
    keys = [get_key(data, keys)]
    return [key for key in keys if key]

@singledispatch
def get_key(data, key):
    return key if key in data else None

@get_keys.register(pd.Series)
def get_key_if_series(data, key):
    return key if key == data.name else None

@get_keys.register(pd.Series)
def get_key_if_series(keys, data):
    return keys if keys == data.name else None


def intersect_keys(keys, X, y=None, **kwargs):
    x_keys = get_keys(keys, X)
    y_keys = get_keys(keys, y)

def split_factory(split_class, split_params=None):
    split_params = split_params if split_params = {}
    return split_class(**split_params)


def default_split_config():
    config = {"split_class": "ShuffleSplit", "split_params": {"random_state: 22"}}


@attr.s(auto_attribs=True)
class Split(BaseEstimator):
    executor: Any = attr.ib(converter=as_factory(split_factory),
                            default=attr.Factory(default_split_config)
                            )
    stratify_keys: Iterable = ()
    group_keys: Iterable = ()

    def get_params(self):
        return attr.asdict(self)

    def get_index_generator(self, *args):
        stratification = intersect_keys(self.stratify_keys, *args) if self.stratify_keys else None
        groups = intersect_keys(self.group_keys, *args) if self.group_keys else None
        index_location_generator = self.executor.split(args[0], stratification, groups)
        return index_location_generator

@attr.s(auto_attribs=True)
class Data(BaseEstimator):
    table: Any
    splitter: Split = attr.ib(converter=as_split, default=attr.Factory(Split))

    target_key: Iterable = None
    prediction_key: Iterable = None
    children: Iterable = attr.Factory(list)

    @classmethod
    def from_table(cls, table, **params):
        return cls(table, **params)

    def from_components(cls, design=None, target=None, prediction=None, **params)       :
        output = []
        target_key = None
        prediction_key = None
        if design is not None:
            output.append(design)
        if target is not None:
            output.append(target)
            target_key = get_key(target)
        if prediction is not None:
            output.append(prediction)
            prediction_key = get_key(prediction)
        if output and len(output) > 1:
            table = pd.concat(output, axis=1)
        elif output:
            table = output[0]

        if target_key:
            params['target_key'] = target_key
        if prediction_key:
            params['prediction_key'] = target_key
        return cls.from_table(table, **params)

    @property
    def index(self):
        return getattr(self.table, 'index', None)

    @property
    def design(self):
        exclusions = append_key(self.target_key, self.prediction_key)
        columns = [i for i in self.table if i not in exclusions]
        return self.table.get(columns)

    @proprety
    def target(self):
        return self.table.get(self.target_key)

    @proprety
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

    def _build_children_split_confs(self):
        heads = self.children[0]
        if isnstance(heads, dict):
            heads = [heads]
        if len(heads) == 1:
            heads = replicate(heads[0], n=2)

        [head.update({'children': None}) for head in heads if 'children' not in head]
        [self.get_params().update(head) for head in heads]
        return heads

    def build_children_confs(self):
        if not self.children:
            confs = replicate(self.get_params(), n=2)
        else:
            confs = self._build_children_split_confs(n=2)
        return confs

    def from_index_locations(self, ilocs, **updates):
        table = self.iloc[ilocs]
        return self.with_params(table=table, **updates)

    def split(self):
        train_conf, test_conf = self.get_children_params()
        generator = self.splitter.get_indices_generator(self.table)
        train_indices, tst_indeices = next(generator)
        train_data = self.from_index_location(**train_conf)
        test_data = self.from_index_location(**test_conf)
        return train_data, test_data

    def append_columns(self, data, location, **updates):
        key = get_key(data)
        location_key = f"{location}_key"
        params = self.get_params()
        params.update(updates)
        params[location_key] = append_key(key, getattr(self, location_key))
        table = pd.concat([self.table, data], axis=1) if self.table is not None else data
        return self.from_table(table, **params)
