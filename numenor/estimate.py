from copy import deepcopy
from functools import cached_property
from hashlib import md5
from typing import *

import numpy as np
import pandas as pd
import variants
from attr import Factory, define, field
from pydantic import BaseModel, create_model
from sklearn import model_selection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from wrapt import decorator

from numenor.transform import Schema
from numenor.utils import (as_factory, call_from_attribute_or_callable,
                           find_value, replace_value)


@decorator
def sklearn_args_from_data(wrapped, instance, args, kwargs):  # type: ignore
    _data = find_value(wrapped, args, kwargs, "data")
    if _data is not None:
        X = _data.features
        y = _data.target
        args, kwargs = replace_value(wrapped, args, kwargs, "data", None)
        args, kwargs = replace_value(wrapped, args, kwargs, "X", X)
        args, kwargs = replace_value(wrapped, args, kwargs, "y", y)
    return wrapped(*args, **kwargs)


class BaseTransformer(BaseEstimator):
    def fit(self, X, y=None, **fit_params):  # type: ignore
        return self

    def transform(self, X):
        return X


@variants.primary
def fit(
    variant,
    estimator,
    *args,
    param_grid=None,
    search_kwargs=None,
    fit_params=None,
    **kwargs,
):
    variant = variant if variant else "base"
    return getattr(fit, variant)(
        estimator,
        *args,
        param_grid=param_grid,
        search_kwargs=search_kwargs,
        fit_params=fit_params,
        **kwargs,
    )


@fit.variant("base")
def fit_base(estimator, *args, fit_params=None, **kwargs):  # type: ignore
    fit_params = fit_params if fit_params is not None else {}
    return estimator.fit(*args, **fit_params)


@fit.variant("sklearn_search")
def fit_sklearn_search(
    estimator, *args, param_grid=None, search_kwargs=None, fit_params=None, **kwargs
):
    search_type = kwargs.get("search_type", "GridSearchCV")
    search_kwargs = search_kwargs if search_kwargs is not None else {}
    fit_params = fit_params if fit_params is not None else {}
    searcher = getattr(model_selection, search_type)(
        estimator, param_grid, **search_kwargs
    )
    return searcher.fit(*args, **fit_params)


@define
class SKAttributeTransformerMixin:
    attribute_name: str = "executor"
    fit_callback: Optional[Union[Callable, str]] = "fit"
    transform_callback: Optional[Union[Callable, str]] = "transform"
    predict_callback: Optional[Union[Callable, str]] = "predict"
    predict_proba_callback: Optional[Union[Callable, str]] = "predict_proba"
    predict_log_proba_callback: Optional[Union[Callable, str]] = "predict_log_proba"
    decision_function_callback: Optional[Union[Callable, str]] = "decision_function"

    @cached_property
    def attribute(self):
        return getattr(self, self.attribute_name)

    def fit(self, X, y=None, **fit_params):
        return call_from_attribute_or_callable(
            self.fit_callback, self.attribute, X, y=y, **fit_params
        )

    def transform(self, X):
        return call_from_attribute_or_callable(
            self.transform_callback, self.attribute, X
        )

    def predict(self, X):
        return call_from_attribute_or_callable(self.predict_callback, self.attribute, X)

    def predict_proba(self, X):
        return call_from_attribute_or_callable(
            self.predict_proba_callback, self.attribute, X
        )

    def predict_log_proba(self, X):
        return call_from_attribute_or_callable(
            self.predict_log_proba_callback, self.attribute, X
        )

    def decision_function(self, X):
        return call_from_attribute_or_callable(
            self.decision_function_callback, self.attribute, X
        )


@define
class Estimator(SKAttributeTransformerMixin, BaseTransformer):
    executor: BaseEstimator = Factory(BaseTransformer)

    response: Callable = field()  # type: ignore

    @response.default
    def response_default(self):
        model = self.executor
        response: str | Callable = "transform"
        while isinstance(model, Pipeline):
            model = model[-1]  # type: ignore
        if isinstance(model, ClassifierMixin):
            response = "predict_proba"
        else:
            response = "predict"
        return response

    feature_schema: Optional[Dict[str, str]] = None
    response_schema: Optional[Dict[str, str] | Dict[str, Dict[str, str]]] = None

    def infer_feature_schema(self, X):
        transformer = self.attribute
        while isinstance(transformer, Pipeline):
            transformer = transformer[0]
        schema = (
            transformer.schema
            if hasattr(transformer, "schema")
            else Schema().fit(X).schema
        )
        features = schema.get("X", {})

        if features:
            self.feature_schema = features

    def infer_response_schema(self, X: np.ndarray | pd.DataFrame):
        prediction = self.respond(X[:10])

        transformer = self.attribute
        while isinstance(transformer, Pipeline):
            transformer = transformer[-1]

        if prediction.ndim < 1:
            return None

        response_type = prediction.dtype.type
        if prediction.ndim == 1 or (prediction.ndim == 2 and prediction.shape[1] == 1):
            schema = {"prediction": response_type}
        else:
            has_classes = hasattr(transformer, "classes_")
            names: List = (
                list(transformer.classes_)
                if has_classes
                else list(range(prediction.shape[1]))
            )
            keys: List[str] = [
                str(i) if has_classes else f"prediction_{i}" for i in names
            ]
            schema = {"prediction": {key: response_type for key in keys}}
        self.response_schema = schema

    def infer_schemas(self, X):
        self.infer_feature_schema(X)
        self.infer_response_schema(X)

    def fit(self, *args, infer_schemas=True, **kwargs):
        super().fit(*args, **kwargs)
        if infer_schemas:
            X = kwargs.get("X", args[0] if args else None)
            self.infer_schemas(X)

    def respond(self, X):
        return call_from_attribute_or_callable(self.response, self.executor, X)


@define
class Trainer(SKAttributeTransformerMixin, BaseTransformer):
    attribute_name: str = "estimator"
    estimator: Any = field(converter=as_factory(Estimator), factory=Estimator)  # type: ignore
    fit_variant: Optional[Union[str, Callable]] = None
    param_grid: Dict[str, Any] = Factory(dict)
    search_kwargs: Dict[str, Any] = Factory(dict)

    id: str = Factory(lambda: md5().hexdigest())

    def _build_kwarg_payload(self, cv, param_grid_updates, search_kwarg_updates):
        param_grid = deepcopy(self.param_grid)
        param_grid.update(param_grid_updates) if param_grid_updates else None

        search_kwargs = deepcopy(self.search_kwargs)
        search_kwargs.update(search_kwarg_updates) if search_kwarg_updates else None
        search_kwargs.update({"cv": cv}) if cv is not None else None
        return param_grid, search_kwargs

    @sklearn_args_from_data  # type: ignore
    def fit(
        self,
        X=None,
        y=None,
        cv=None,
        data=None,
        param_grid_updates=None,
        search_kwarg_updates=None,
        fit_variant=None,
        **fit_params,
    ):
        if data is not None:
            raise ValueError("data should be converted, to (X, y)")
        param_grid, search_kwargs = self._build_kwarg_payload(
            cv, param_grid_updates, search_kwarg_updates
        )

        fit_variant = fit_variant if fit_variant else self.fit_variant
        fit(
            self.fit_variant,
            self.estimator,
            X,
            y,
            param_grid=param_grid,
            search_kwargs=search_kwargs,
            **fit_params,
        )
        return self
