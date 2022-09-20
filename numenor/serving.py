from attrs import define, field, Factory
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from functools import singledispatch


@singledispatch
def package_predictions(model, prediction):
    if isinstance(prediction, np.ndarray) and prediction.shape == (1, ):
        prediction = prediction[0]
    return {'prediction': prediction}


@package_predictions.register(ClassifierMixin)
def package_predictions(model, prediction):
    classes = model.classes_
    if isinstance(
            prediction,
            np.ndarray) and prediction.ndim == 2 and prediction.shape[0] == 1:
        probabilities = {
            class_: prediction[0][idx]
            for idx, class_ in enumerate(classes)
        }

    if isinstance(
            prediction,
            np.ndarray) and prediction.ndim == 2 and prediction.shape[0] == 1:
        prediction = classes[prediction[0, :].argmax()]

    return {
        'probabilities': probabilities,
        'argmax_class': classes[prediction[0, :]],
    }


def pandas_feature_converter(features):
    return pd.DataFrame.from_dict(features, orient='index')


@define
class Serving:
    transformer: BaseEstimator
    feature_schema: Dict = field()

    @feature_schema.default
    def default_feature_schema(self):
        ...

    response_schema: Dict = field()

    @response_schema.default
    def default_response_schema(self):
        ...

    feature_converter: Callable = pandas_feature_converter
    respone: Callable = package_predictions

    def __call__(self, features, *args, **kwargs):
        features = self.feature_converter(features)
        return self.response(transformer, features, *args, **kwargs)

    @classmethod
    def infer_from_transformer(
            cls,
            transformer,
            instance_check: Callable = lambda x: isinstance(x, Pipeline),
            **extras):
        model = transformer.executor
        while instance_check(model):
            model = model[-1]
        return cls.infer_from_model(model)

    @classmethod
    def infer_from_model(cls, model, **extras):
        ...
        schema = {'model_id': str}

    def from_serving_response(self, **extras):
        schema = self.schema.copy()
        schema.update(extras)
        return evolve(self, schema=schema)
