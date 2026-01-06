import requests
import time

def test_service_health():
    """Service is up and responding (Smoke Test)"""
    # API'nin kalkması için kısa bir bekleme
    url = "http://localhost:8000/"
    
    # 5 deneme yaparak servisin kalkmasını bekle (Resilient test)
    for _ in range(5):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                assert response.json()["status"] == "healthy"
                return
        except:
            time.sleep(2)
            
    assert False, "Servis zamanında ayağa kalkmadı!"

def test_single_prediction():
    """Send a single prediction request to verify service"""
    url = "http://localhost:8000/predict"
    payload = {
        "tenure": 12, "MonthlyCharges": 70.5, "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check", "gender": "Female",
        "service_combo_id": "Fiber optic_Yes_No_No", "geo_code": "G25"
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()