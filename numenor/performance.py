from attr import define, field, Factory
from enum import Enum
from typing import *
from numenor.utils import get_attribute_or_call, get_signature_params
from functools import singledispatch


def default_data_keys():
    return {
        'y_true': 'target',
        'y_score': 'prediction',
        'y_pred': 'prediction'
    }


@define
class Metric:
    executor: Callable
    data_keys: Dict[str, str] = Factory(default_data_keys)
    name: str = field()

    @name.default
    def name_default(self):
        if hasattr(self.executor, '__name__'):
            name = self.executor.__name__
        elif hasattr(self.executor, '__class__'):
            name = self.executor.__class__.__name__
        else:
            name = 'unnamed_metric'
        return name

    call_kwargs: Dict = Factory(dict)
    with_estimator: bool = False
    infer_from_signature: bool = True

    def __call__(self, data, estimator=None):
        payload = {
            **{
                key: get_attribute_or_call(accessor, data)
                for key, accessor in self.data_keys.items()
            },
            **({
                'estimator': estimator
            } if self.with_estimator else {})
        }
        if self.infer_from_signature:
            params = get_signature_params(self.executor)
            payload = {
                key: value
                for key, value in payload.items() if key in params
            }
        return self.executor(**{**payload, **self.call_kwargs})


@singledispatch
def _as_dict(metrics):
    return metrics


@_as_dict.register(list)
def _as_dict_from_list(metrics):
    return {metric.name: metric for metric in metrics}


@define
class Performance:
    metrics: List[Metric] = field(converter=_as_dict, default=Factory(list))
    evaluation_kwargs: Dict = Factory(dict)
    group_key: str = None

    def evaluate(self, data):
        return {
            metric.name: metric(data, estimator)
            for metric in self.metrics
        }

    def evaluate_groups(self, data, group_key=None):
        group_key = group_key if group_key else self.group_key
        groups = data.groupby(group_key)
        grouped_evaluation = {
            _key: self.evaluate(_data)
            for _key, _data in groups
        }
        return grouped_evaluation
