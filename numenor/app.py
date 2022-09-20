import os
from typing import Optional

import cloudpickle
from fastapi import FastAPI
from pydantic import create_model

predictor = FastAPI()


def load_prediction_model(location: Optional[str]):
    if location is None:
        raise ValueError("no default location for model available")

    with open(location, "rb") as f:
        prediction_model = cloudpickle.load(f)
    return prediction_model


PREDICTION_MODEL = load_prediction_model(location=os.getenv("model_path"))

Features = create_model(
    "Features",
    **{
        feature: (Optional[_klass], ...)
        for feature, _klass in PREDICTION_MODEL.schema.items()
    }
)

Response = create_model(
    "Response",
    **{
        feature: (Optional[_klass], ...)
        for feature, _klass in PREDICTION_MODEL.response.items()
    }
)


@predictor.post("/predict_probabilities", response_model=Response)
async def predict(id: int, features: Features, **kwargs):
    return PREDICTION_MODEL.serve(features, id=id)
