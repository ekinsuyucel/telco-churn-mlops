from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

app = FastAPI(title="Telco Churn Model API - MLOps Level 2")

# Modelin yükleneceği yol (Kişi 3'ün eğittiği model buraya gelecek)
MODEL_PATH = "models/model.joblib"

@app.get("/")
def health_check():
    """Hocanın istediği sistem sağlığı kontrolü"""
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "service": "Stateless Serving Function"
    }

@app.post("/predict")
def predict(input_data: dict):
    """
    Hocanın istediği Stateless Serving Function (REST endpoint).
    High-cardinality feature'lar (geo_code, service_combo_id) bu JSON içinde gelir.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model henüz hazır değil!")

    try:
        # JSON verisini DataFrame'e çevir
        df = pd.DataFrame([input_data])
        
        # Modeli yükle ve tahmin et
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]

        return {
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(probability[0]),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tahmin hatası: {str(e)}")