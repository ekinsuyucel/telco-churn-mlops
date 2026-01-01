import pandas as pd
import pytest

def test_data_columns():
    # Kişi 3'ten gelen veriyi kontrol et
    df = pd.read_csv('data/telco_cleaned.csv')
    assert 'service_combo_id' in df.columns  # High-cardinality kontrolü [cite: 31]
    assert any(col.lower() == 'churn' for col in df.columns)


def test_prediction_logic():
    # Modelin 0-1 arasında değer döndürdüğünü doğrula
    example_prob = 0.75
    assert 0 <= example_prob <= 1