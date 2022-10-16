from enum import Enum, auto
from functools import singledispatch
from typing import *

import pandas as pd
from attrs import Factory, define, field
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.utils import Bunch

from numenor.utils import (call_from_attribute_or_callable,
                           get_attribute_or_call, get_signature_params)


def default_data_keys():
    return {
        "y_true": "target",
        "y_score": "prediction",
        "y_pred": "prediction",
        "probas_pred": "prediction",
    }


def default_view(*args, **kwargs):
    try:
        return display(*args, **kwargs)  # type: ignore
    except NameError:
        return print(*args, **kwargs)


def plot(value, *args, **kwargs):
    return pd.Series(value[0], index=value[1]).plot(*args, **kwargs)


class RenderKind(Enum):
    SCALAR = auto()
    SERIES = auto()
    DATAFRAME = auto()
    KEY_VALUE = auto()
    ARRAY = auto()
    PLOT = auto()


@define
class Measurement:
    name: str
    value: Any
    view: Callable
    render_kind: Any = RenderKind.SCALAR
    view_args: Tuple = Factory(tuple)
    view_kwargs: Dict = Factory(dict)
    verbose: bool = True

    def render(self):
        if self.verbose:
            print(self.name)
        return call_from_attribute_or_callable(
            self.view, self.value, *self.view_args, **self.view_kwargs
        )


@define
class Metric:
    executor: Callable
    data_keys: Dict[str, str] = Factory(default_data_keys)
    name: str = field()

    @name.default
    def name_default(self):
        if hasattr(self.executor, "__name__"):
            name = self.executor.__name__
        elif hasattr(self.executor, "__class__"):
            name = self.executor.__class__.__name__
        else:
            raise ValueError("metrics require name")
        return name

    view: Callable = default_view
    render_kind: Any = RenderKind.SCALAR
    call_kwargs: Dict = Factory(dict)
    with_estimator: bool = False
    infer_from_signature: bool = True

    def __call__(self, data, estimator=None):
        payload = {
            **{
                key: get_attribute_or_call(accessor, data)
                for key, accessor in self.data_keys.items()
            },
            **({"estimator": estimator} if self.with_estimator else {}),
        }
        if self.infer_from_signature:
            params = get_signature_params(self.executor)
            payload = {key: value for key, value in payload.items() if key in params}
        value = self.executor(**{**payload, **self.call_kwargs})
        return Measurement(self.name, value, self.view, self.render_kind)


@singledispatch
def _as_name(name):
    return str(name)


@_as_name.register(str)
def _as_name_str(name):
    return name


@_as_name.register(tuple)
@_as_name.register(list)
def _as_name_collection(name):
    return "__".join(name)


def f1_curve(y_true, probas_pred, *, pos_label=None, sample_weight=None):
    p, r, t = metrics.precision_recall_curve(
        y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight
    )
    f1 = 2 * p * r / (p + r)
    return f1[:-1], t


CLASSIFICATION_METRICS = [
    Metric(metrics.roc_auc_score, render_kind=RenderKind.SCALAR),
    Metric(metrics.average_precision_score, render_kind=RenderKind.SCALAR),
    Metric(
        metrics.PrecisionRecallDisplay.from_predictions,
        name="precision_recall_curve",
        view="plot",
        render_kind=RenderKind.PLOT,
    ),
    Metric(f1_curve, view=plot, render_kind=RenderKind.PLOT),
]


@define
class Performance:
    metrics: List[Metric] = Factory(list)
    evaluation_kwargs: Dict = Factory(dict)
    group_keys: List[str] = Factory(list)
    report: Any = None

    def evaluate(self, data, estimator=None):
        evaluation = {metric.name: metric(data, estimator) for metric in self.metrics}
        return Bunch(**evaluation)

    def evaluate_group(self, data, group_key=None):
        groups = data.groupby(group_key)
        grouped_evaluation = {
            _as_name(_key): self.evaluate(_data) for _key, _data in groups
        }
        return Bunch(**grouped_evaluation)

    def evaluate_groups(self, data, group_keys=None):
        group_keys = group_keys if group_keys else self.group_keys
        return {
            _as_name(group): self.evaluate_group(data, group) for group in group_keys
        }

    def build_report(self, data, estimator=None):
        topline_evaluation = self.evaluate(data, estimator)
        group_evaluations = self.evaluate_groups(data, group_keys) if group_keys else {}
        self.report = {"top_line": topline_evaluation, **group_evaluations}
        return self.report

    # !!!TODO group measures
    def collate_group(self, measurements, group, metric_name, view="auto"):
        top_line = measurements["top_line"][metric_name]
        data = {
            key: value[metric_name].value for key, value in measurements[group].items()
        }
        if view == "auto":
            if top_line.render_kind is RenderKind.SCALAR:
                series = pd.concat(
                    [
                        pd.Series([top_line], index=["top_line"], name=metric_name),
                        pd.Series(data, name=metric_name),
                    ]
                )
                series.index.name = "group"
                top_line.render(series.to_frame())
            if top_line.render_kind is RenderKind.PLOT:
                fig, ax = plt.subplots()
                legend = []
                top_line.render(ax=ax)
                for group, value in data.items():
                    legend.append(group)
                    top_line.render(value, ax=ax, verbose=False)
                plt.legend(legend)
