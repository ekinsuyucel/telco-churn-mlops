from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

from app.schemas import PredictRequest

app = FastAPI(title="Telco Churn Model API - MLOps Level 2")

MODEL_PATH = "models/model.joblib"


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "service": "Stateless Serving Function"
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model henüz hazır değil!")

    try:
        df = pd.DataFrame([request.dict()])

        model = joblib.load(MODEL_PATH)
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]

        return {
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(probability[0]),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
