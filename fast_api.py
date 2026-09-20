from pathlib import Path
from typing import Literal
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_PATH = Path(os.getenv("MODEL_PATH"))
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
@app.get("/last_update_date")   
def last_update_date():
    try:
        last_update = MODEL_PATH.stat().st_mtime
        date = datetime.fromtimestamp(last_update).strftime("%d-%m-%Y")
        print(date)
        return {"last_update_date": date}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve last update date: {str(e)}"
        )    