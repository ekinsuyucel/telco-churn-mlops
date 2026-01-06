from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str
    PaymentMethod: str
    gender: str
    service_combo_id: str
    geo_code: str
    # Opsiyonel alanlar (Varsayılan değerlerle)
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    PaperlessBilling: str = "Yes"