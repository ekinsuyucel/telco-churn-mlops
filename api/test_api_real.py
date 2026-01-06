from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_health():
    """Sistem sağlığı kontrolü testi"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_prediction_payload():
    """ZORUNLU TEST: Yeni şemaya göre tahmin doğrulaması"""
    test_payload = {
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "gender": "Female",
        "service_combo_id": "Fiber optic_Yes_No_No", # Kişi 1 formatı
        "geo_code": "G25"
    }
    response = client.post("/predict", json=test_payload)
    
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert "churn_probability" in response.json()

def test_predict_invalid_data():
    """Hata yönetimi: Yanlış veri tipi gönderildiğinde 422 dönmeli"""
    response = client.post("/predict", json={"tenure": "on iki"}) # int yerine string
    assert response.status_code == 422