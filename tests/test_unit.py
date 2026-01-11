import pytest
import pandas as pd
from app.utils import HashingTransformer

def test_hashing_consistency():
   
    transformer = HashingTransformer(cols=["geo_code"], n_features=1024)
    data = pd.DataFrame({"geo_code": ["G1", "G1", "G2"]})
    
    transformed = transformer.fit_transform(data)
    
   
    assert (transformed[0].toarray() == transformed[1].toarray()).all()

    assert not (transformed[0].toarray() == transformed[2].toarray()).all()

def test_hashing_output_shape():
   
    n_feat = 512
    transformer = HashingTransformer(cols=["service_id"], n_features=n_feat)
    data = pd.DataFrame({"service_id": ["A", "B"]})
    transformed = transformer.fit_transform(data)
    
    assert transformed.shape[1] == n_feat
