from typing import *

import cloudpickle
import pandas as pd
import yaml
from fire import Fire
from numpy.typing import ArrayLike
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

from numenor import data as d
from numenor import estimate as e
from numenor import pipeline as p
from numenor import transform as t

TRANSFORMER_REGISTRY = {
    "schema": t.Schema,
    "simpleimputer": SimpleImputer,
    "xgbclassifier": XGBClassifier,
}


def load_sklearn_data(dataset):
    X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
    y = pd.Series(dataset["target"], name="label")
    return X, y


def load_dataset():
    X, y = load_sklearn_data(datasets.load_iris())

    dataset = d.Dataset.from_method(
        features=X,
        target=y,
        splitter={"executor": StratifiedShuffleSplit(), "stratify_keys": ["label"]},
    )
    return dataset


def get_fit_params(pipeline, params, validation_dataset: List[Tuple[ArrayLike]]):
    params.update({"eval_set": [tuple(validation_dataset)]})
    return p.package_params(pipeline, params)


def get_pipeline():
    return p.make_pipeline(
        SimpleImputer(),
        XGBClassifier(n_estimators=10000, max_depth=4, eval_metric="mlogloss"),
    )


def get_estimator(pipeline):
    return e.Estimator(executor=pipeline)


def get_trainer(validation_dataset):
    pipeline = get_pipeline()
    fit_params = get_fit_params(
        pipeline,
        {"early_stopping_rounds": 20},
        validation_dataset,
    )
    return e.Trainer(estimator=get_estimator(pipeline), fit_params=fit_params)


def main():
    dataset = load_dataset()
    train_dataset, evaluation_dataset = dataset.split()
    fitting_dataset, validation_dataset = train_dataset.split()

    trainer = get_trainer(validation_dataset)
    trainer.fit(data=train_dataset)


if __name__ == "__main__":
    Fire(main)
