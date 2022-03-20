import attr
import variants
from wrapt import decorator

from sklearn.base import BaseEstimator, clone
from typing import *
from numenor.utils import find_value, replace_value, enforce_first_arg
from functools import singledispatch

@enforce_first_arg
@singledispatch
def get_attribute_or_call(attribute_or_callable, obj, args, kwargs):
    return attribute_or_callable(obj, *args, **kwargs)

@get_attribute_or_call.register(str)
def get_attribute_or_call_(attribute_or_callable, obj, args, kwargs):
    return getattr(obj, attribute_or_callable)(*args, **kwargs)

@enforce_first_arg
@singledispatch
def get_from_registry_or_call(key_or_callable, registry, args, kwargs):
    return key_or_callable(obj, *args, **kwargs)

@get_attribute_or_call.register(str)
def get_from_registry_or_call(key_or_callable, registry, args, kwargs):
    return regsitry[key](*args, **kwargs)


@decorator
def sklearn_from_data(wrapped, instance, args, kwargs):
    data = find_value(wrapped, args, kwargs, 'data')
    if data is not None:
        X = data.features
        y = data.target
    args, kwargs = replace_values(wrapped, args, kwargs, 'data', None)
    args, kwargs = replace_values(wrapped, args, kwargs, 'X', X)
    args, kwargs = replace_values(wrapped, args, kwargs, 'y', y)
    return wrapped(*args, **kwargs)


class BaseTransformer(BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X


@variants.primary
def parameter_search(variant, estimator, param_grid=None, search_kwargs=None, fit_params=None, **kwargs):
    variant = variant if variant else 'base'
    return getattr(parameter_search, variant)(estimator, **kwargs)

@parameter_search.variant('base')
def parameter_search_base(estimator, param_grid, *args, search_kwargs=None, **fit_params):
    return estimator.fit(*args, **kwargs)

@parameter_search.variant('sklearn')
def parameter_search_sklearn(estimator, param_grid, *args, search_kwargs=None, **fit_params):
    search_type = kwargs.get('search_type', 'GridSearchCV')
    search_kwargs = search_kwargs if search_kwargs is not None else {}
    searcher = getattr(model_selection, search_type)(estimator, param_grid, **search_kwargs)
    return searcher.fit(*args, **fit_params)


@attr.s(auto_attribs=True)
class BaseTransform(BaseTransformer):
    prediction_callback: Optional[Union[Callable, str]] = 'predict'
    transform_callback: Optional[Union[Callable, str]] = 'transform'
    decision_function_callback: Optional[Union[Callable, str]] = 'decision_function'

    def transform(self, X):
        return get_attribute_or_call(self.prediction_callback, self.estimator_, X)

    def predict(self, X):
        return get_attribute_or_call(self.transform_callback, self.estimator_, X)

    def decision_function(self, X):
        return get_attribute_or_call(self.decision_function_callback, self.estimator_, X)


@attr.s(auto_attribs=True)
class Transform(BaseTransformer):
    executor: BaseEstimator
    parameter_optimization: Optional[Union[str, Callable]] = None
    param_grid: Dict[str, Any] = attr.Factory(dict)
    search_kwargs: Dict[str, Any] =  attr.Factory(dict)

    @sklearn_from_data
    def fit(self, X, y=None, cv=None, data=None, **fit_params):
        self.search_kwargs.update({'cv': cv}) if cv is not None else None
        self.estimator_ = parameter_search(self.parameter_optimization, estimator, self.param_grid, X, y, search_kwargs=self.search_kwargs, **fit_params) 
        return self


# class TransformRegressor(ClassifierMixin, BaseTransformer):
    # pass


# class TransformClassifier(ClassifierMixin, BaseTransformer):
    # def predict_proba(self, X):
        # return get_attribute_or_call(prediction_callback, self.estimator_, X)

    # def predict_log_proba(self, X):
        # return get_attribute_or_call(prediction_callback, self.estimator_, X)
