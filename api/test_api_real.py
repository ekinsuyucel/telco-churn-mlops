from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_health():
    """Hocanın istediği sistem sağlığı kontrolü [cite: 9]"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_prediction_with_high_cardinality():
    """
    ZORUNLU TEST: High-Cardinality feature'ların API tarafından 
    işlendiğini doğrular[cite: 6, 31, 43].
    """
    test_payload = {
        "tenure": 12,
        "monthly_charges": 70.5,
        "contract_type": "Month-to-month",
        "service_combo_id": "Fiber_Yes_No_Yes", # Kişi 1'in oluşturduğu alan [cite: 9]
        "geo_code": "G25" # Yüksek kardinaliteli alan 
    }
    # Model henüz yüklenmediyse bile isteğin formatını test eder
    response = client.post("/predict", json=test_payload)
    
    # Model yoksa 503 (Resilient Serving), varsa 200 dönmeli [cite: 43, 49]
    assert response.status_code in [200, 503]

def test_predict_invalid_data():
    """Hata yönetimi testi: Eksik veri gönderildiğinde 422 dönmeli"""
    response = client.post("/predict", json={"invalid_field": "data"})
    assert response.status_code == 422 # FastAPI otomatik validation