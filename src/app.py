from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np
import os


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failure using XGBoost",
    version="1.0.0"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load(
    os.path.join(MODEL_DIR, "model.pkl")
)

with open(
    os.path.join(MODEL_DIR, "feature_names.json"),
    "r"
) as f:
    feature_names = json.load(f)

with open(
    os.path.join(MODEL_DIR, "threshold.json"),
    "r"
) as f:
    threshold_data = json.load(f)

threshold = float(threshold_data["threshold"])


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class PredictionInput(BaseModel):

    volt: float
    rotate: float
    pressure: float
    vibration: float

    hour: int
    day: int
    month: int
    dayofweek: int

    volt_mean_6h: float
    volt_std_6h: float
    rotate_mean_6h: float
    rotate_std_6h: float
    pressure_mean_6h: float
    pressure_std_6h: float
    vibration_mean_6h: float
    vibration_std_6h: float

    volt_change_1h: float
    rotate_change_1h: float
    pressure_change_1h: float
    vibration_change_1h: float

    volt_mean_24h: float
    volt_std_24h: float
    rotate_mean_24h: float
    rotate_std_24h: float
    pressure_mean_24h: float
    pressure_std_24h: float
    vibration_mean_24h: float
    vibration_std_24h: float

    volt_change_24h: float
    rotate_change_24h: float
    pressure_change_24h: float
    vibration_change_24h: float


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Predictive Maintenance API is running",
        "model": "XGBoost",
        "features": len(feature_names),
        "threshold": threshold
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: PredictionInput):

    input_data = data.model_dump()

    # Make sure features are in exactly
    # the same order as during training
    values = [
        input_data[feature]
        for feature in feature_names
    ]

    X = np.array(values).reshape(1, -1)

    probability = float(
        model.predict_proba(X)[0][1]
    )

    prediction = (
        "FAILURE"
        if probability >= threshold
        else "NO FAILURE"
    )

    return {
        "failure_probability": round(probability, 4),
        "threshold": threshold,
        "prediction": prediction
    }