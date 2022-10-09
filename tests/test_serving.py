from numenor import estimate as e
from numenor import serving as s
from numenor import transform as t


def test_serving_classes():
    estimator = e.Estimator(executor=t.BaseTransformerNumpy(), response="transform")  # type: ignore
    serve = s.Serve(estimator)
    data = {"a": 1, "b": 2}
    estimator.fit(serve.feature_converter(data))
    response = serve(data)
    assert response == {"predicted_class": "b", "probabilities": data}


def test_serving_classes():
    estimator = e.Estimator(executor=t.BaseTransformerNumpy(), response="transform")  # type: ignore
    data = {"a": 1, "b": 2}
    serve = s.Serve(estimator)
    estimator.fit(serve.feature_converter(data))

    response = serve(data)
    assert response == {"predicted_class": "b", "probabilities": data}

    data = {"a": 1, "b": 2}
    estimator = e.Estimator(executor=t.BaseTransformerNumpy(), response="transform")  # type: ignore
    serve = s.Serve(estimator)
    response = serve(data)
    assert response == {"prediction": {"0": 1, "1": 2}}
