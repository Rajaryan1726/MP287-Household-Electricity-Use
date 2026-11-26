import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.mp287.logging_config import get_logger

app = FastAPI(title="MP287 Electricity API")
logger = get_logger()
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load("models/model.joblib")
    return _model


class PredictRequest(BaseModel):
    rooms: int
    members: int
    ac_hours_per_day: float = Field(..., le=25)
    avg_temp_c: float = Field(..., ge=4)
    house_area_sqft: float
    solar_kw: float
    connection: str
    feeder: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.model_dump()])
    pred = float(get_model().predict(df)[0])
    logger.info("prediction", extra={"extra_data": {"feeder": req.feeder, "prediction": pred}})
    return {"prediction": pred}