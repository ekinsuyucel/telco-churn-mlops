from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
import sys
from app.utils import HashingTransformer
from app.schemas import PredictRequest

# Joblib'in sınıfı bulabilmesi için __main__'e bağlıyoruz
sys.modules['__main__'].HashingTransformer = HashingTransformer

app = FastAPI(title="Telco Churn Prediction API")
MODEL_PATH = "models/model.joblib"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ Model başarıyla yüklendi!")
        except Exception as e:
            print(f"❌ Yükleme hatası: {e}")

@app.get("/")
def health():
    return {"status": "healthy", "model_ready": model is not None}

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model hazır değil.")
    try:
        df = pd.DataFrame([request.dict()])
        
        # Feature Crosses
        df["cross_contract_payment"] = df["Contract"].astype(str) + "_x_" + df["PaymentMethod"].astype(str)
        df["cross_service_contract"] = df["service_combo_id"].astype(str) + "_x_" + df["Contract"].astype(str)
        df["cross_geo_contract"] = df["geo_code"].astype(str) + "_x_" + df["Contract"].astype(str)

        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]

        return {
            "churn_prediction": int(prediction[0]),
            "churn_probability": round(float(probability[0]), 4),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))