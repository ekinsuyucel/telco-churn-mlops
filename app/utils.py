from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher

class HashingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_features=2**14):
        self.cols = cols
        self.n_features = n_features
        
    def fit(self, X, y=None):
        self.hasher_ = FeatureHasher(n_features=self.n_features, input_type="string")
        return self
        
    def transform(self, X):
        # API ve Training'de ortak kullanılan dönüşüm mantığı
        tokens = [[f"{c}={val}" for c, val in zip(self.cols, row)] for row in X[self.cols].values]
        return self.hasher_.transform(tokens)