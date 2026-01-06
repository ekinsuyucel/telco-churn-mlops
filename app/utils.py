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
        # --- PART 3: SABOTAGE START ---
        # Ödev gereği bilerek hata fırlatıyoruz (The Sabotage)
        # Bu satır CI pipeline'daki Unit Test aşamasını patlatacaktır.
        raise ValueError("STOP THE LINE: Sabotaj testi için bilerek hata fırlatıldı!") 
        # --- PART 3: SABOTAGE END ---

        # Aşağıdaki kod normalde çalışan kısımdır ama yukarıdaki hata yüzünden çalışmayacaktır
        tokens = [[f"{c}={val}" for c, val in zip(self.cols, row)] for row in X[self.cols].values]
        return self.hasher_.transform(tokens)