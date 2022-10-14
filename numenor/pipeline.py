from __future__ import annotations

from functools import singledispatch
from typing import *

import variants
from attrs import define, field
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, _final_estimator_has, _name_estimators
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if


@singledispatch
def _access(accessor, steps):
    name, transform = steps[accessor]
    return name, transform, accessor


@_access.register(list)
def _access_list(accessor, steps):
    head, tail = accessor[0], accessor[1:]
    name, transform = steps[head]
    return name, transform, tail if tail else head


def package_params(
    pipeline: Pipeline,
    params: Dict,
    accessor: Hashable = -1,
    instance_check: Callable = lambda x: isinstance(x, Pipeline),
) -> Dict:
    name, transform, accessor = _access(accessor, pipeline.steps)
    if instance_check(transform):
        packaged_params = package_params(
            transform, params, accessor=accessor, instance_check=instance_check
        )
        updated_params = {
            f"{name}__{key}": value for key, value in packaged_params.items()
        }
    else:
        updated_params = {f"{name}__{key}": value for key, value in params.items()}
    return updated_params


@variants.primary
def transform_fit_params(
    variant: str = "key",
    pipeline: Pipeline = None,
    key: Hashable = "eval_set",
    fit_params: Dict = None,
    **kwargs,
) -> Dict:
    return getattr(transform_fit_params, variant)(
        pipeline=pipeline, key=key, fit_params=fit_params, **kwargs
    )


@transform_fit_params.variant("base")
def transform_fit_params_base(pipeline, fit_params, **kwargs) -> Dict:
    return fit_params


@transform_fit_params.variant("key")
def transform_fit_params_key(pipeline, key, fit_params, **kwargs) -> Dict:
    if key in fit_params:
        transformed_eval_sets = []
        for X, y in fit_params[key]:
            Xt = X  # sklearn convention
            for _, name, transform in pipeline._iter(with_final=False):
                Xt = transform.transform(Xt)
            transformed_eval_sets.append((Xt, y))
        fit_params[key] = transformed_eval_sets
    return fit_params


class Pipeline(Pipeline):
    def __init__(
        self, steps, memory=None, verbose=False, sampler=None, fit_params_variant="key"
    ):
        if sampler is not None and not hasattr(sampler, "fit_resample"):
            raise TypeError(f"sampler: {sampler} is not a valid resampler")
        self.sampler = sampler
        self.fit_params_variant = fit_params_variant
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **fit_params) -> PipelineXGBResample:
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        if self.sampler is not None:
            Xt, y = self.sampler.fit_resample(Xt, y)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                fit_params_last_step = transform_fit_params(
                    variant=self.fit_params_variant,
                    pipeline=self,
                    fit_params=fit_params_last_step,
                )
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        if sampler is not None:
            Xt, y = self.sampler.fit_resample(Xt, y)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]

            fit_params_last_step = transform_fit_params(
                variant=self.fit_params_variant,
                pipeline=self,
                fit_params=fit_params_last_step,
            )
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(Xt, y, **fit_params_last_step)
            else:
                return last_step.fit(Xt, y, **fit_params_last_step).transform(Xt)

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        if sampler is not None:
            Xt, y = self.sampler.fit_resample(Xt, y)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            fit_params_last_step = transform_fit_params(
                variant=self.fit_params_variant,
                pipeline=self,
                fit_params=fit_params_last_step,
            )
            y_pred = self[-1].fit_predict(Xt, y, **fit_params_last_step)
        return y_pred


def make_pipeline(
    *steps, memory=None, verbose=False, sampler=None, fit_params_variant="key"
) -> Pipeline:
    return Pipeline(
        _name_estimators(steps),
        memory=memory,
        verbose=verbose,
        sampler=sampler,
        fit_params_variant=fit_params_variant,
    )
