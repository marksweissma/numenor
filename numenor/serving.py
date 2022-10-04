from functools import singledispatch
from typing import *

import pandas as pd
import variants
from attrs import define, field
from numpy.typing import ArrayLike
from pydantic import BaseModel
from sklearn.base import BaseEstimator

from numenor.estimate import Estimator


@variants.primary
def package_prediction(variant="infer", model=None, prediction=None, **kwargs):
    return getattr(package_prediction, variant)(model, prediction, **kwargs)


@package_prediction.variant("infer")
def package_prediction_infer(model: BaseEstimator, prediction: ArrayLike):
    if prediction.ndim == 1 or prediction.ndim == 2 and prediction.shape[1] == 1:  # type: ignore
        result = package_prediction.one_dimensional(model, prediction)

    elif prediction.ndim == 2:  # type: ignore
        result = package_prediction.two_dimensional(model, prediction)
    else:
        result = {"prediction": prediction}


@package_prediction.variant("one_dimensional")
def package_prediction_one_dimensional(model: BaseEstimator, prediction: ArrayLike):  # type: ignore
    prediction = prediction[0] if prediction.ndim == 1 else prediction[0, 0]  # type: ignore
    return {"prediction": prediction}


@package_prediction.variant("two_dimensional")
@singledispatch
def package_prediction_two_dimensional(model: BaseEstimator, prediction: ArrayLike):
    if hasattr(model, "classes_"):
        keys = model.classes_  # type: ignore
        key_name = "probabilities"
        probabilities = {class_: prediction[0, idx] for idx, class_ in enumerate(classes)}  # type: ignore
        prediction = classes[prediction[0, :].argmax()]  # type: ignore
        payload = {
            "probabilities": probabilities,
            "prediction": prediction,
        }
    else:
        payload = {"prediction": prediction}
    return payload


def pydantic_to_pandas_feature_converter(features: BaseModel):
    return pd.DataFrame.from_dict(features.dict(), orient="index")


@define
class Serve:
    estimator: Estimator
    feature_schema: Dict
    response_schema: Dict
    feature_converter: Callable = pydantic_to_pandas_feature_converter
    response: Callable = package_prediction

    def __call__(self, features, *args, **kwargs):
        features = self.feature_converter(features)
        prediction = self.estimator.response(features)
        return self.response(self.estimator, prediction, *args, **kwargs)
