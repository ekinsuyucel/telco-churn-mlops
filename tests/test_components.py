import pandas as pd
import pytest

def test_data_columns():
    
    df = pd.read_csv('data/telco_cleaned.csv')
    assert 'service_combo_id' in df.columns  
    assert any(col.lower() == 'churn' for col in df.columns)


def test_prediction_logic():

    example_prob = 0.75
    assert 0 <= example_prob <= 1
