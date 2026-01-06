import pytest
import pandas as pd
from app.utils import HashingTransformer

def test_hashing_consistency():
    """Hashing her zaman aynı girdi için aynı sonucu vermeli (İzole Unit Test)"""
    transformer = HashingTransformer(cols=["geo_code"], n_features=1024)
    data = pd.DataFrame({"geo_code": ["G1", "G1", "G2"]})
    
    transformed = transformer.fit_transform(data)
    
    # İlk iki satır (G1) aynı hash değerine sahip olmalı
    assert (transformed[0].toarray() == transformed[1].toarray()).all()
    # Farklı girdi (G2) farklı (veya en azından rastgele değil, tutarlı) olmalı
    assert not (transformed[0].toarray() == transformed[2].toarray()).all()

def test_hashing_output_shape():
    """Çıktı boyutu n_features ile eşleşmeli"""
    n_feat = 512
    transformer = HashingTransformer(cols=["service_id"], n_features=n_feat)
    data = pd.DataFrame({"service_id": ["A", "B"]})
    transformed = transformer.fit_transform(data)
    
    assert transformed.shape[1] == n_feat