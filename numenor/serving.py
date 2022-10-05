from functools import singledispatch
from typing import *

import pandas as pd
import variants
from attrs import define
from numpy.typing import ArrayLike
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

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
    return result


@package_prediction.variant("one_dimensional")
def package_prediction_one_dimensional(model: BaseEstimator, prediction: ArrayLike):  # type: ignore
    prediction = prediction[0] if prediction.ndim == 1 else prediction[0, 0]  # type: ignore
    return {"prediction": prediction}


@package_prediction.variant("two_dimensional")
@singledispatch
def package_prediction_two_dimensional(model: BaseEstimator, prediction: ArrayLike):
    while isinstance(model, Pipeline):
        model = model[-1]
    if hasattr(model, "classes_"):
        keys = model.classes_  # type: ignore
        probabilities = {class_: prediction[0, idx] for idx, class_ in enumerate(keys)}  # type: ignore
        prediction = keys[prediction[0, :].argmax()]  # type: ignore
        payload = {
            "probabilities": probabilities,
            "prediction": prediction,
        }
    else:
        payload = {
            "prediction": {idx: value for idx, value in enumerate(prediction[0, :])}
        }
    return payload


def pydantic_to_pandas_feature_converter(features: BaseModel):
    return pd.DataFrame.from_dict(features.dict(), orient="index")


@define
class Serve:
    estimator: Estimator
    feature_converter: Callable = pydantic_to_pandas_feature_converter
    response: Callable = package_prediction
    response_variant: str = "infer"

    def __call__(self, request_model, *args, **kwargs):
        features = self.feature_converter(request_model)
        prediction = self.estimator.respond(features)
        return self.response(
            self.response_variant,
            model=getattr(self.estimator, "attribute", self.estimator),
            prediction=prediction,
            *args,
            **kwargs
        )
