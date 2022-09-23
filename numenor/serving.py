from typing import *

import pandas as pd
import variants
from attrs import define, field
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from numenor.estimate import Transformer


@variants.primary
def package_prediction(variant="infer", model=None, prediction=None, **kwargs):
    return getattr(package_prediction, variant)(model, prediction, **kwargs)


@package_prediction.variant("infer")
def package_prediction_infer(model: BaseEstimator, prediction: ArrayLike):
    if prediction.ndim == 1 or prediction.ndim == 2 and prediction.shape[1] == 1:
        result = package_prediction.one_dimensional(model, prediction)

    elif prediction.ndim == 2:
        result = package_prediction.two_dimensional(model, prediction)
    else:
        result = {"prediction": prediction}


@package_prediction.variant("one_dimensional")
def package_prediction_one_dimensional(model: BaseEstimator, prediction: ArrayLike):
    prediction = prediction[0] if prediction.ndim == 1 else prediction[0, 0]
    return {"prediction": prediction}


@package_prediction.variant("two_dimensional")
def package_prediction_two_dimensional(model: ClassifierMixin, prediction: ArrayLike):
    classes = model.classes_
    probabilities = {class_: prediction[0, idx] for idx, class_ in enumerate(classes)}
    prediction = classes[prediction[0, :].argmax()]

    return {
        "probabilities": probabilities,
        "argmax_class": prediction,
    }


def pandas_feature_converter(features):
    return pd.DataFrame.from_dict(features, orient="index")


@define
class Serve:
    transformer: Transformer
    feature_schema: Dict
    response_schema: Dict = field()
    feature_converter: Callable = pandas_feature_converter
    response: Callable = package_prediction

    def __call__(self, features, *args, **kwargs):
        features = self.feature_converter(features)
        prediction = self.transformer.serve(features)
        return self.response(model=self.transformer, prediction, *args, **kwargs)

    @classmethod
    def infer_from_transformer_and_example(
        cls,
        transformer,
        instance_check: Callable = lambda x: isinstance(x, Pipeline),
        **extras
    ):
        if 'feature_schema' not in extras and hastattr(transformer, 'feature_schema'):
            extras['feature_schema'] = transformer.feature_schema
        if 'response_schema' not in extras and hastattr(transformer, 'feature_schema'):
            extras['response_schema'] = transformer.feature_schema


    @classmethod
    def infer_from_model(cls, model, **extras):
