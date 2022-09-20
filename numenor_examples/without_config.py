from functools import lru_cache

import pandas as pd
import structlog
import variants
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier, XGBRegressor

from numenor import data, estimate
from numenor.pipeline import make_pipeline, package_params

LOG = structlog.get_logger()
RANDOM_STATE = 42


@lru_cache(None)
def load_sklearn_dataset(handle):
    dataset = handle()
    X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
    y = pd.Series(dataset["target"], name="label")
    return X, y


@variants.primary
def regression_example(variant=None, **kwargs):
    variant = variant if variant else "base"
    return getattr(regression_example, variant)(**kwargs)


@regression_example.variant("base")
def regression_example_base(plot=True, **kwargs):
    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    model = make_pipeline(RobustScaler(), Lasso())
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    LOG.msg(
        "regression performance",
        r2_score=r2_score(y_test, predictions),
        mean_absolute_error=mean_absolute_error(y_test, predictions),
    )
    if plot:
        plt.scatter(y_test, predictions)
        plt.show()


@regression_example.variant("transformer")
def regression_example_transformer(plot=True, **kwargs):
    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    model = estimate.Transformer(make_pipeline(RobustScaler(), Lasso()))
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    LOG.msg(
        "regression performance",
        r2_score=r2_score(y_test, predictions),
        mean_absolute_error=mean_absolute_error(y_test, predictions),
    )
    if plot:
        plt.scatter(y_test, predictions)
        plt.show()


@regression_example.variant("pipeline_with_early_stopping")
def regression_example_pipeline_with_early_stopping(**kwargs):

    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    # Using numnenor's pipeline wrapper enables support for preprocessing transformers
    # and samplers before fitting without hacks / boilerplate / duplication
    model = make_pipeline(
        RobustScaler(),
        KMeans(n_clusters=3),
        XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.3),
    )

    # split training data into "fitting" and "validating" for early stopping
    X_train_fit, X_train_validate, X_train_fit, y_train_validate = train_test_split(
        X_train, y_train
    )

    early_stopping_params = {
        "xgbregressor__early_stopping_rounds": 5,
        "xgbregressor__eval_metric": ["rmse"],
        "xgbregressor__eval_set": [(X_train_validate, y_train_validate)],
    }

    model.fit(X_train, y_train, **early_stopping_params)

    predictions = model.predict(X_test)

    LOG.msg(
        "regression performance",
        r2_score=r2_score(y_test, predictions),
        mean_absolute_error=mean_absolute_error(y_test, predictions),
    )


@regression_example.variant("data__column")
def regression_example_data__column(**kwargs):

    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    metadata = pd.Series(["metadata"] * len(X), name="metadata")
    full_data = data.Data.from_components(features=X, target=y, metadata=metadata)

    model = make_pipeline(RobustScaler(), Lasso())
    model.fit(full_data.features, full_data.target)
    LOG.msg("All columns available", columns=full_data.columns)
    LOG.msg("Features", features=full_data.feature_keys)
    LOG.msg(
        "metadata",
        metadata={
            "name": full_data.metadata.name,
            "type": full_data.metadata.__class__,
        },
    )


@regression_example.variant("data__column_name")
def regression_example_data__column_name(**kwargs):

    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    metadata = pd.Series(["my_metadata"] * len(X), name="metadata")
    table = pd.concat([X, y, metadata], axis=1)

    full_data = data.Data(table, target_column="label", metadata_column="metadata")

    model = make_pipeline(RobustScaler(), Lasso())
    model.fit(full_data.features, full_data.target)
    LOG.msg("All columns available", columns=full_data.columns)
    LOG.msg("Features", features=full_data.feature_keys)
    LOG.msg(
        "metadata",
        metadata={
            "name": full_data.metadata.name,
            "type": full_data.metadata.__class__,
        },
    )


@regression_example.variant("dataset")
def regression_example_dataset(**kwargs):

    X, y = load_sklearn_dataset(datasets.load_diabetes)

    full_dataset = data.Dataset.from_method("from_components", features=X, target=y)

    model = make_pipeline(RobustScaler(), Lasso())

    train_dataset, test_dataset = full_dataset.split()

    model.fit(train_dataset.features, train_dataset.target)

    predictions = model.predict(test_dataset.features)

    LOG.msg(
        "regression performance",
        r2_score=r2_score(test_dataset.target, predictions),
        mean_absolute_error=mean_absolute_error(test_dataset.target, predictions),
    )


@regression_example.variant("dataset__early_stopping")
def regression_example_dataset__early_stopping(**kwargs):

    X, y = load_sklearn_dataset(datasets.load_diabetes)

    full_dataset = data.Dataset.from_method("from_components", features=X, target=y)

    model = make_pipeline(
        RobustScaler(),
        KMeans(n_clusters=3),
        XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.3),
    )

    train_dataset, test_dataset = full_dataset.split()
    train_dataset_fitting, train_dataset_validating = train_dataset.split()

    fit_params = package_params(
        model,
        {
            "early_stopping_rounds": 5,
            "eval_metric": ["rmse"],
            "eval_set": [
                (train_dataset_validating.features, train_dataset_validating.target)
            ],
        },
    )

    model.fit(train_dataset.features, train_dataset.target, **fit_params)

    predictions = model.predict(test_dataset.features)

    LOG.msg(
        "dataset lengths",
        full=len(full_dataset),
        train=len(train_dataset),
        train_fitting=len(train_dataset_fitting),
        train_validating=len(train_dataset_validating),
    )

    LOG.msg(
        "regression performance",
        r2_score=r2_score(test_dataset.target, predictions),
        mean_absolute_error=mean_absolute_error(test_dataset.target, predictions),
    )
