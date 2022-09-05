from attrs import define, field, Factory
from functools import singledispatch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

import itertools
import variants

import numpy as np
import pandas as pd

from numenor.utils import as_factory, call_from_attribute_or_callable

from typing import *


class PandasMixin:

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            columns = list(X)
            index = X.index
        Xt = super().transform(X)
        if isinstance(X, pd.DataFrame):
            Xt = pd.DataFrame(Xt, index=index, columns=columns)
        return Xt


class BaseTransformer(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X


@variants.primary
def include(variant, df, **kwargs):
    return getattr(include, variant)(df, **kwargs)


@include.variant('by_key')
def include_by_key(df, include=None, **kwargs):
    keys = [i for i in df if i in include] if include is not None else list(df)
    return keys


@include.variant('by_dtype')
def include_by_key(df, use_head=True, **kwargs):
    selected = (df.head() if use_head else df).select_dtypes(**kwargs)
    keys = [i for i in df if i in selected]
    return keys


@variants.primary
def exclude(variant, df, exclude, **kwargs):
    return getattr(exclude, variant)(df, exclude, **kwargs)


@exclude.variant('by_key')
def exclude_by_key(df, exclude, **kwargs):
    keys = [i for i in df if i not in exclude]
    return keys


@exclude.variant('by_dtype')
def exclude_by_key(df, use_head=True, **kwargs):
    if use_head:
        df = df.head()
    keys = [i for i in df if i not in list(df.select_dtypes(**kwargs))]
    return keys


@define
class Select:
    include: Dict = Factory(dict)
    exclude: Dict = Factory(dict)

    def __call__(self, df):
        if not self.include:
            includes = list(df)
        else:
            includes = list(
                itertools.chain(*(
                    include(inclusion, df, **kwargs)
                    for inclusion, kwargs in self.include.items())))
        if self.exclude:
            excludes = set(
                itertools.chain(*(
                    exclude(exclusion, df, **kwargs)
                    for exclusion, kwargs in self.exclude.items())))
        else:
            excludes = set()
        keys = [i for i in includes if i not in excludes]
        return df[keys]


@singledispatch
def infer_schema(obj):
    raise TypeError(f'obj: {obj} type: {type(obj)} not supported')


@infer_schema.register(pd.DataFrame)
def infer_schema_df(obj):
    schema = {}
    [schema.update(infer_schema(obj[column])) for column in obj]
    return schema


@infer_schema.register(pd.Series)
def infer_schema_series(obj):
    first_valid_value = obj.loc[obj.first_valid_index()]
    return {obj.name: type(first_valid_value)}


@define
class SchemaSelectMixin:
    selection: Callable = field(converter=as_factory(Select),
                                default=Factory(dict))
    infer_schema: Callable = infer_schema

    def build_schema(self, **kwargs):
        for key, value in kwargs.items():
            self.schema[key] = self.infer_schema(value)
        return self.schema

    def fit(self, X, y=None, **fit_params):
        self.schema = {}
        X_t = self.selection(X)
        payload = {'X': X_t}
        payload.update({'y': y}) if y is not None else None

        self.build_schema(**payload)
        return super().fit(X, y=y, **fit_params)

    def transform(self, X):
        X_t = X[self.schema['X']]
        return super().transform(X_t)


@define
class Schema(SchemaSelectMixin, PandasMixin, BaseTransformer):
    ...


class SimpleImputer(PandasMixin, SimpleImputer):
    pass


class RobustScaler(PandasMixin, RobustScaler):
    pass


class StandardScaler(PandasMixin, StandardScaler):
    pass


SECONDS_IN_DAY = 24 * 3600
SECONDS_IN_WEEK = 7 * 24 * 3600


@define
class CallableTransformer(BaseTransformer):
    attr: Union[str, Callable] = 'fillna'
    attr_args: Tuple = ('missing', )
    attr_kwargs: Dict = Factory(dict)

    def transform(self, X):
        return call_from_attribute_or_callable(self.attr, X, *self.attr_args,
                                               **self.attr_kwargs)


@define
class TimeFeaturesProjectUnitCircle(TransformerMixin, BaseEstimator):
    columns: List
    deltas: List = field()
    unit: str = '1D'

    @deltas.default
    def default_deltas(self):
        return list(itertools.combinations(self.columns, 2))

    def single_column_features(self, df, column):
        second_of_day = df[column].dt.hour * 3600 + df[
            column].dt.minute * 60 + df[column].dt.second
        second_of_week = df[
            column].dt.dayofweek * SECONDS_IN_DAY + second_of_day
        df[f'{column}_time_of_day_x'] = np.cos(2 * np.pi * second_of_day /
                                               SECONDS_IN_DAY)
        df[f'{column}_time_of_day_y'] = np.sin(2 * np.pi * second_of_day /
                                               SECONDS_IN_DAY)
        df[f'{column}_time_of_week_x'] = np.cos(2 * np.pi * second_of_week /
                                                SECONDS_IN_WEEK)
        df[f'{column}_time_of_week_y'] = np.sin(2 * np.pi * second_of_week /
                                                SECONDS_IN_WEEK)
        return df

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        for column in self.columns:
            X[column] = pd.to_datetime(X[column])
            X = self.single_column_features(X, column)

        for start, stop in self.deltas:
            X[f'{start}__{stop}_delta'] = (X[stop] - X[start]) / pd.Timedelta(
                self.unit)
        columns = set(self.columns).union(chain.from_iterable(self.deltas))
        return X.drop(columns, axis=1)
