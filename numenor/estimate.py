import attr
import variants
from wrapt import decorator
from copy import deepcopy

from sklearn.base import BaseEstimator, clone
from sklearn import model_selection
from typing import *
from numenor.utils import find_value, replace_value, enforce_first_arg
from functools import singledispatch


@enforce_first_arg
@singledispatch
def get_attribute_or_call(attribute_or_callable, obj, *args, **kwargs):
    return attribute_or_callable(obj, *args, **kwargs)


@get_attribute_or_call.register(str)
def get_attribute_or_call_(attribute_or_callable, obj, *args, **kwargs):
    return getattr(obj, attribute_or_callable)(*args, **kwargs)


@enforce_first_arg
@singledispatch
def get_from_registry_or_call(key_or_callable, registry, *args, **kwargs):
    return key_or_callable(obj, *args, **kwargs)


@get_from_registry_or_call.register(str)
def get_from_registry_or_call(key_or_callable, registry, *args, **kwargs):
    return regsitry[key](*args, **kwargs)


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
def parameter_search(variant,
                     estimator,
                     *args,
                     param_grid=None,
                     search_kwargs=None,
                     fit_params=None,
                     **kwargs):
    variant = variant if variant else 'base'
    return getattr(parameter_search, variant)(estimator,
                                              *args,
                                              param_grid=param_grid,
                                              search_kwargs=search_kwargs,
                                              fit_params=fit_params,
                                              **kwargs)


@parameter_search.variant('base')
def parameter_search_base(estimator, *args, fit_params=None, **kwargs):
    fit_params = fit_params if fit_params is not None else {}
    return estimator.fit(*args, **fit_params)


@parameter_search.variant('sklearn')
def parameter_search_base(estimator,
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


@attr.s(auto_attribs=True)
class Transformer(BaseTransformer):
    executor: BaseEstimator
    parameter_optimization: Optional[Union[str, Callable]] = None
    param_grid: Dict[str, Any] = attr.Factory(dict)
    search_kwargs: Dict[str, Any] = attr.Factory(dict)

    transform_callback: Optional[Union[Callable, str]] = 'transform'
    predict_callback: Optional[Union[Callable, str]] = 'predict'
    predict_proba_callback: Optional[Union[Callable, str]] = 'predict_proba'
    predict_log_proba_callback: Optional[Union[Callable,
                                               str]] = 'predict_log_proba'
    decision_function_callback: Optional[Union[Callable,
                                               str]] = 'decision_function'

    def _build_kwarg_payload(self, cv, param_grid_updates,
                             search_kwarg_updates):
        param_grid = deepcopy(self.param_grid)
        param_grid.update(
            search_kwarg_updates) if search_kwarg_updates else None

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
            **fit_params):
        param_grid, search_kwargs = self._build_kwarg_payload(
            cv, param_grid_updates, search_kwarg_updates)
        self.estimator_ = parameter_search(self.parameter_optimization,
                                           self.executor,
                                           X,
                                           y,
                                           param_grid=self.param_grid,
                                           search_kwargs=search_kwargs,
                                           **fit_params)
        return self

    def transform(self, X):
        return get_attribute_or_call(self.transform_callback, self.estimator_,
                                     X)

    def predict(self, X):
        return get_attribute_or_call(self.predict_callback, self.estimator_, X)

    def predict_proba(self, X):
        return get_attribute_or_call(self.predict_proba_callback,
                                     self.estimator_, X)

    def predict_log_proba(self, X):
        return get_attribute_or_call(self.predict_log_proba_callback,
                                     self.estimator_, X)

    def decision_function(self, X):
        return get_attribute_or_call(self.decision_function_callback,
                                     self.estimator_, X)


# class TransformRegressor(ClassifierMixin, BaseTransformer):
# pass

# class TransformClassifier(ClassifierMixin, BaseTransformer):
# def predict_proba(self, X):
# return get_attribute_or_call(prediction_callback, self.estimator_, X)

# def predict_log_proba(self, X):
# return get_attribute_or_call(prediction_callback, self.estimator_, X)
