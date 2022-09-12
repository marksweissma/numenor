import cloudpickle
import uvicorn
from fastapi import FastAPI

predictor = FastAPI()


def load_prediction_model(location):
    with open(location, 'rb') as f:
        model = cloudpickle.load(f)
    return Predictor(model)


PREDICION_MODEL = load_prediction_model(location=os.getenv('model_path'))

Features = create_model(
    'Features', **{
        feature: (Optional[_klass], ...)
        for feature, _klass in PREDICTION_MODEL.schema.items()
    })

Response = create_model(
    'Response', **{
        feature: (Optional[_klass], ...)
        for feature, _klass in PREDICTION_MODEL.response.items()
    })


@predictor.post("/predict_probabilities", response_model=Response)
async def predict(id: int, features: Features, **kwargs):
    return PREDICTION_MODEL.serve(features, id=id)
