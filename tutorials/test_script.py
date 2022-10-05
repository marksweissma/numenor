from functools import lru_cache, singledispatch

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from numenor import estimate, serving
from numenor.transform import Schema

RANDOM_STATE = 42


@lru_cache(None)
def load_sklearn_dataset(handle):
    dataset = handle()
    X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
    dataset = handle()
    y = pd.Series(dataset["target"], name="label")
    return X, y


def test_regression_example():
    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    model = estimate.Trainer(
        estimator={"executor": make_pipeline(Schema(), RobustScaler(), Lasso())}
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    response_schema = Schema().fit(predictions)

    serve = serving.Serve(model.estimator, feature_converter=lambda x: x)
    result = serve(X.iloc[[0]])
    assert isinstance(result.item(), float)


def test_regression_example():
    X, y = load_sklearn_dataset(datasets.load_iris)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    model = estimate.Trainer(
        estimator={
            "executor": make_pipeline(Schema(), RobustScaler(), LogisticRegression())
        }
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    response_schema = Schema().fit(predictions)

    serve = serving.Serve(model.estimator, feature_converter=lambda x: x)
    result = serve(X.iloc[[0]])
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert isinstance(result["probabilities"][0], float)
    assert isinstance(result["prediction"], np.integer)
