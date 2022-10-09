import itertools
from functools import singledispatch
from typing import *

import numpy as np
import pandas as pd
import variants
from attrs import Factory, define, field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.preprocessing import RobustScaler, StandardScaler  # type: ignore

from numenor.utils import as_factory, call_from_attribute_or_callable


class PandasMixin:
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            columns = list(X)
            index = X.index
        Xt = super().transform(X)  # type: ignore
        if isinstance(X, pd.DataFrame):
            Xt = pd.DataFrame(Xt, index=index, columns=columns)  # type: ignore
        return Xt


class BaseTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X


class BaseTransformerNumpy(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        if hasattr(X, "columns"):
            self.classes_ = list(X.columns)
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return X


@variants.primary
def select_include(variant, df, **kwargs):
    return getattr(select_include, variant)(df, **kwargs)


@select_include.variant("by_key")
def select_include_by_key(df, include=None, **kwargs):
    keys = [i for i in df if i in include] if include is not None else list(df)
    return keys


@select_include.variant("by_dtype")
def select_include_by_dtype(df, use_head=True, **kwargs):
    selected = (df.head() if use_head else df).select_dtypes(**kwargs)
    keys = [i for i in df if i in selected]
    return keys


@variants.primary
def select_exclude(variant, df, **kwargs):
    return getattr(select_exclude, variant)(df, **kwargs)


@select_exclude.variant("by_key")
def select_exclude_by_key(df, exclude, **kwargs):
    keys = [i for i in df if i not in exclude]
    return keys


@select_exclude.variant("by_dtype")
def select_exclude_by_type(df, use_head=True, **kwargs):
    selected = (df.head() if use_head else df).select_dtypes(**kwargs)
    keys = [i for i in df if i not in selected]
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
                itertools.chain(
                    *(
                        select_include(method, df, **kwargs)
                        for method, kwargs in self.include.items()
                    )
                )
            )
        if self.exclude:
            exclusions = (
                select_exclude(method, df, **kwargs)
                for method, kwargs in self.exclude.items()
            )
            excludes = set(itertools.chain(*(i for i in exclusions if i)))
        else:
            excludes = set()
        keys = [i for i in includes if i not in excludes]
        return df[keys]


@singledispatch
def infer_schema(obj, *args, **kwargs):
    raise TypeError(f"obj: {obj} type: {type(obj)} not supported")


@infer_schema.register(pd.DataFrame)
def infer_schema_df(obj, *args, **kwargs):
    schema = {}
    [schema.update(infer_schema(obj[column])) for column in obj]
    return schema


@infer_schema.register(pd.Series)
def infer_schema_series(obj, *args, **kwargs):
    first_valid_value = obj.loc[obj.first_valid_index()]
    return {obj.name: type(first_valid_value)}


@infer_schema.register(np.ndarray)
def infer_schema_ndarray(obj, *args, **kwargs):
    if obj.ndim == 2:
        schema = {idx: str(obj.dtype.type) for idx in range(obj.shape[1])}
    elif obj.ndim == 1:
        schema = {0: str(obj.dtype.type)}
    else:
        raise ValueError("only 1D and 2D ndarrays supported")
    return schema


@define
class SchemaSelectMixin:
    selection: Callable = field(converter=as_factory(Select), default=Factory(dict))  # type: ignore
    infer_schema: Callable = infer_schema

    def build_schema(self, **kwargs):
        for key, value in kwargs.items():
            self.schema[key] = self.infer_schema(value)
        return self.schema

    def fit(self, X, y=None, **fit_params):
        self.schema = {}
        if isinstance(X, np.ndarray):
            X_t = X
        else:

            X_t = self.selection(X)
        payload = {"X": X_t}
        payload.update({"y": y}) if y is not None else None

        self.build_schema(**payload)
        return super().fit(X, y=y, **fit_params)  # type: ignore

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_t = X[self.schema["X"]]
        else:
            X_t = X
        return super().transform(X_t)  # type: ignore


@define
class Schema(SchemaSelectMixin, PandasMixin, BaseTransformer):
    ...


class SimpleImputer(PandasMixin, SimpleImputer):
    pass


class RobustScaler(PandasMixin, RobustScaler):
    pass


class StandardScaler(PandasMixin, StandardScaler):
    pass


@define
class CallableTransformer(BaseTransformer):
    attr: Union[str, Callable] = "fillna"
    attr_args: Tuple = ("missing",)
    attr_kwargs: Dict = Factory(dict)

    def transform(self, X):
        return call_from_attribute_or_callable(
            self.attr, X, *self.attr_args, **self.attr_kwargs
        )


SECONDS_IN_DAY = 24 * 3600
SECONDS_IN_WEEK = 7 * 24 * 3600


@define
class TimeFeaturesProjectUnitCircle(TransformerMixin, BaseEstimator):
    columns: List
    deltas: List = field()
    unit: str = "1D"

    @deltas.default  # type: ignore
    def default_deltas(self):
        return list(itertools.combinations(self.columns, 2))

    def single_column_features(self, df, column):
        second_of_day = (
            df[column].dt.hour * 3600 + df[column].dt.minute * 60 + df[column].dt.second
        )
        second_of_week = df[column].dt.dayofweek * SECONDS_IN_DAY + second_of_day
        df[f"{column}_time_of_day_x"] = np.cos(
            2 * np.pi * second_of_day / SECONDS_IN_DAY
        )
        df[f"{column}_time_of_day_y"] = np.sin(
            2 * np.pi * second_of_day / SECONDS_IN_DAY
        )
        df[f"{column}_time_of_week_x"] = np.cos(
            2 * np.pi * second_of_week / SECONDS_IN_WEEK
        )
        df[f"{column}_time_of_week_y"] = np.sin(
            2 * np.pi * second_of_week / SECONDS_IN_WEEK
        )
        return df

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        for column in self.columns:
            X[column] = pd.to_datetime(X[column])
            X = self.single_column_features(X, column)

        for start, stop in self.deltas:
            X[f"{start}__{stop}_delta"] = (X[stop] - X[start]) / pd.Timedelta(self.unit)
        columns = set(self.columns).union(itertools.chain.from_iterable(self.deltas))
        return X.drop(columns, axis=1)
