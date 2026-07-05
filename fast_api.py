from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = Path("models/heart_pipeline.joblib")
model = joblib.load(MODEL_PATH)


class PredictBody(BaseModel):
    age: int
    sex: Literal[0, 1]
    chest_pain_type: Literal[1, 2, 3, 4]
    bp: int
    cholesterol: int
    fbs_over_120: Literal[0, 1]
    ekg_results: Literal[0, 1, 2]
    max_hr: int
    exercise_angina: Literal[0, 1]
    st_depression: float
    slope_st: Literal[1, 2, 3]
    nr_of_vessels: Literal[0, 1, 2, 3]
    thallium: Literal[3, 6, 7]


@app.get("/")
def health_check():
    return {"status": "running"}


@app.post("/predict")
def predict(body: PredictBody):
    try:
        X = pd.DataFrame([{
            "Age": body.age,
            "Sex": body.sex,
            "Chest": body.chest_pain_type,
            "BP": body.bp,
            "Chol": body.cholesterol,
            "FBS over 120": body.fbs_over_120,
            "EKG": body.ekg_results,
            "Max HR": body.max_hr,
            "Exercise angina": body.exercise_angina,
            "Depr": body.st_depression,
            "Slope of ST": body.slope_st,
            "Vessels": body.nr_of_vessels,
            "Thallium": body.thallium,
        }])

        probability = model.predict_proba(X)[0, 1]

        return {
            "heart_disease_probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )