import pytest
from numenor import pipeline as p
import numpy as np
from sklearn.base import BaseEstimator
from numpy import testing as npt


class AddOne(BaseEstimator):

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X):
        return X + 1


class FitParamServer(BaseEstimator):

    def fit(self, X, y=None, **fit_params):
        self.fit_params = fit_params
        return self

    def transform(self, X):
        return X


@pytest.fixture
def pipeline():
    return p.make_pipeline(AddOne(), FitParamServer())


@pytest.fixture
def X():
    return np.arange(10).reshape(10, 1)


@pytest.fixture
def fit_params(X):
    return {'eval_set': [(X + 10, None)]}


# TODO: parameterize these
def test_package_params(pipeline, fit_params):
    updated_fit_params = p.package_params(pipeline, fit_params.copy())
    assert 'fitparamserver__eval_set' in updated_fit_params

    updated_fit_params = p.package_params(
        pipeline,
        fit_params.copy(),
        accessor=0,
    )
    assert 'addone__eval_set' in updated_fit_params


def test_transform_params(pipeline, X, fit_params):
    fit_params = p.transform_fit_params('base', fit_params=fit_params.copy())
    assert fit_params == fit_params
    transformed_fit_params = p.transform_fit_params(
        'key',
        pipeline,
        fit_params=fit_params.copy(),
    )
    X_original = fit_params['eval_set'][0][0]
    X_t = transformed_fit_params['eval_set'][0][0]
    npt.assert_almost_equal(X_original + 1, X_t)


def test_pipeline(pipeline, X, fit_params):

    X_original = fit_params['eval_set'][0][0]
    fit_params = p.package_params(pipeline, fit_params.copy())
    pipeline.fit(X, y=None, **fit_params)
    X_t = pipeline[-1].fit_params['eval_set'][0][0]
    npt.assert_almost_equal(X_original + 1, X_t)
