from attr import define, field, Factory
import variants
from wrapt import decorator
from copy import deepcopy

from sklearn.base import BaseEstimator, clone
from sklearn import model_selection
from typing import *
from numenor.utils import find_value, replace_value, enforce_first_arg, call_from_attribute_or_callable
from functools import singledispatch
from hashlib import md5


@decorator
def sklearn_args_from_data(wrapped, instance, args, kwargs):
    _data = find_value(wrapped, args, kwargs, 'data')
    if _data is not None:
        X = _data.features
        y = _data.target
        args, kwargs = replace_value(wrapped, args, kwargs, 'data', None)
        args, kwargs = replace_value(wrapped, args, kwargs, 'X', X)
        args, kwargs = replace_value(wrapped, args, kwargs, 'y', y)
    return wrapped(*args, **kwargs)


class BaseTransformer(BaseEstimator):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X


@variants.primary
def fit(variant,
        estimator,
        *args,
        param_grid=None,
        search_kwargs=None,
        fit_params=None,
        **kwargs):
    variant = variant if variant else 'base'
    return getattr(fit, variant)(estimator,
                                 *args,
                                 param_grid=param_grid,
                                 search_kwargs=search_kwargs,
                                 fit_params=fit_params,
                                 **kwargs)


@fit.variant('base')
def fit_base(estimator, *args, fit_params=None, **kwargs):
    fit_params = fit_params if fit_params is not None else {}
    return estimator.fit(*args, **fit_params)


@fit.variant('sklearn_search')
def fit_sklearn_search(estimator,
                       *args,
                       param_grid=None,
                       search_kwargs=None,
                       fit_params=None,
                       **kwargs):
    search_type = kwargs.get('search_type', 'GridSearchCV')
    search_kwargs = search_kwargs if search_kwargs is not None else {}
    fit_params = fit_params if fit_params is not None else {}
    searcher = getattr(model_selection, search_type)(estimator, param_grid,
                                                     **search_kwargs)
    return searcher.fit(*args, **fit_params)


@define
class Transformer(BaseTransformer):
    executor: BaseEstimator
    fit_variant: Optional[Union[str, Callable]] = None
    param_grid: Dict[str, Any] = Factory(dict)
    search_kwargs: Dict[str, Any] = Factory(dict)

    transform_callback: Optional[Union[Callable, str]] = 'transform'
    predict_callback: Optional[Union[Callable, str]] = 'predict'
    predict_proba_callback: Optional[Union[Callable, str]] = 'predict_proba'
    predict_log_proba_callback: Optional[Union[Callable,
                                               str]] = 'predict_log_proba'
    decision_function_callback: Optional[Union[Callable,
                                               str]] = 'decision_function'
    id: str = Factory(lambda: md5().hexdigest())

    def _build_kwarg_payload(self, cv, param_grid_updates,
                             search_kwarg_updates):
        param_grid = deepcopy(self.param_grid)
        param_grid.update(param_grid_updates) if param_grid_updates else None

        search_kwargs = deepcopy(self.search_kwargs)
        search_kwargs.update(
            search_kwarg_updates) if search_kwarg_updates else None
        search_kwargs.update({'cv': cv}) if cv is not None else None
        return param_grid, search_kwargs

    @sklearn_args_from_data
    def fit(self,
            X=None,
            y=None,
            cv=None,
            data=None,
            param_grid_updates=None,
            search_kwarg_updates=None,
            fit_variant=None,
            **fit_params):
        if data is not None:
            raise ValueError('data should be converted, to (X, y)')
        param_grid, search_kwargs = self._build_kwarg_payload(
            cv, param_grid_updates, search_kwarg_updates)
        fit_variant = fit_variant if fit_variant else self.fit_variant
        self.estimator_ = fit(self.fit_variant,
                              self.executor,
                              X,
                              y,
                              param_grid=param_grid,
                              search_kwargs=search_kwargs,
                              **fit_params)
        return self

    def transform(self, X):
        return call_from_attribute_or_callable(self.transform_callback,
                                               self.estimator_, X)

    def predict(self, X):
        return call_from_attribute_or_callable(self.predict_callback,
                                               self.estimator_, X)

    def predict_proba(self, X):
        return call_from_attribute_or_callable(self.predict_proba_callback,
                                               self.estimator_, X)

    def predict_log_proba(self, X):
        return call_from_attribute_or_callable(self.predict_log_proba_callback,
                                               self.estimator_, X)

    def decision_function(self, X):
        return call_from_attribute_or_callable(self.decision_function_callback,
                                               self.estimator_, X)
