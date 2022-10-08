import os
from typing import Dict, Optional, Type

import cloudpickle
import structlog
from attrs import Factory, define
from fastapi import FastAPI, status
from pydantic import BaseModel, create_model

from numenor.estimate import Estimator
from numenor.serving import Serve

LOG = structlog.get_logger()

predictor = FastAPI()


@define
class ServingHook:
    serve: Serve
    feature_model: Type[BaseModel]
    response_model: Type[BaseModel]
    response_extras: Dict[str, Type] = Factory(dict)

    @staticmethod
    def create_nullable_value_model(schema: Dict[str, Type], name: str, **extras):
        model: Type[BaseModel] = create_model(
            name,
            **{
                **{
                    feature: (Optional[_klass], ...)
                    for feature, _klass in schema.items()
                },
                **extras,
            },
        )
        return model

    @staticmethod
    def load_estimator_from_location(location: Optional[str] = None):

        defaulted_location = location if location else os.getenv("model_path")
        assert defaulted_location, "no provided default location for model available"

        with open(defaulted_location, "rb") as estimator_location:
            estimator: Estimator = cloudpickle.load(estimator_location)
        return estimator

    @classmethod
    def from_location(cls, location: Optional[str] = None, **response_extras):
        estimator = cls.load_estimator_from_location(location)

        assert estimator.feature_schema, "estimator has no feature schema"
        assert estimator.response_schema, "estimator has no response schema"

        feature_model = cls.create_nullable_value_model(
            estimator.feature_schema, "Features"
        )
        response_model = cls.create_nullable_value_model(
            estimator.response_schema, "Response", **response_extras
        )
        return cls(Serve(estimator), feature_model, response_model, response_extras)

    def update_estimator(self, estimator: Estimator):
        try:
            assert estimator.feature_schema, "estimator has no feature schema"
            assert estimator.response_schema, "estimator has no response schema"
            feature_model = self.create_nullable_value_model(
                estimator.feature_schema, "Features"
            )
            response_model = self.create_nullable_value_model(
                estimator.response_schema, "Response", **self.response_extras
            )
            assert feature_model == self.feature_model
            assert response_model == self.response_model
            self.serve.set_estimator(estimator)
            success: bool = True

        except Exception as e:
            success: bool = False

        return success


Predictor = ServingHook.from_location(id=int)


@predictor.post("/predict", response_model=Predictor.response_model)
async def predict(
    features: Predictor.feature_model, id: Optional[int] = None, **kwargs  # type: ignore
):
    response = Predictor.serve(features, id=id)


@predictor.post(
    "/update_estimator",
    status_code=status.HTTP_202_ACCEPTED | status.HTTP_406_NOT_ACCEPTABLE,
)
async def update_estimator(location: Optional[str]):
    try:
        estimator = ServingHook.load_estimator_from_location(location)
        response = status.HTTP_202_ACCEPTED

    except Exception as e:
        LOG.failure(f"failed to load estimator")
        response = status.HTTP_406_NOT_ACCEPTABLE

    return response
