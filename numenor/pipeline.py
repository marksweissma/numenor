from sklearn.pipeline import Pipeline, _final_estimator_has
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if

from functools import singledispatch


def package_terminal_estimator_params(pipeline, params, instance_check=lambda x: hasattr(x, 'steps')):
    name, transform = pipeline.steps[-1]
    if instance_check(transform):
        packaged_params = package_terminal_estimator_params(transform, params, instance_check)
        updated_params = {f'{name}__{key}': value for key, value in packaged_params.items()}
    else:
        updated_params = {f'{name}__{key}': value for key, value in params.items()}
    return updated_params


def transform_xgb_eval_set_if_in_fit_params(pipeline, **fit_params):
    if 'eval_set' in fit_params:
        transformed_eval_sets = []
        for X, y in fit_params['eval_set']:
            Xt = X  # sklearn convention
            for _, name, transform in pipeline._iter(with_final=False):
                Xt = transform.transform(Xt)
            transformed_eval_sets.append((Xt, y))
        fit_params['eval_set'] = transformed_eval_sets
    return fit_params


class Pipeline(Pipeline):

    def fit(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                fit_params_last_step = transform_xgb_eval_set_if_in_fit_params(self, fit_params_last_step)
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            fit_params_last_step = transform_xgb_eval_set_if_in_fit_params(self, fit_params_last_step)
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(Xt, y, **fit_params_last_step)
            else:
                return last_step.fit(Xt, y, **fit_params_last_step).transform(Xt)

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            fit_params_last_step = transform_xgb_eval_set_if_in_fit_params(self, fit_params_last_step)
            y_pred = self.steps[-1][1].fit_predict(Xt, y, **fit_params_last_step)
        return y_pred
