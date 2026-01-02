from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    # ZORUNLU (testin kullandıkları)
    tenure: int
    monthly_charges: float
    contract_type: str

    # High-cardinality feature'lar
    service_combo_id: str
    geo_code: str

    # Opsiyonel (training tarafında olabilir)
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
