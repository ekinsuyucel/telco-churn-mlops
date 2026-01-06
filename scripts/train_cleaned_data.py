import os
import sys
import yaml
import pandas as pd
import joblib
from pathlib import Path

# --- KRİTİK: PROJE DİZİNİNİ TANITMA ---
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from app.utils import HashingTransformer  # Artık hata vermez
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

class CleanedDataTrainer:
    def __init__(self):
        self.project_root = Path(root_path)
        config_path = self.project_root / "config" / "cleaned_data_config.yaml"
        
        if not config_path.exists():
            self.config = {"data": {"source": "data/telco_cleaned.csv"}, "random_state": 42}
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

    def run(self):
        print("📥 Veri yükleniyor...")
        df = pd.read_csv(self.config["data"]["source"])
        df['churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        print("🔀 Feature crosses oluşturuluyor...")
        df["cross_contract_payment"] = df["Contract"].astype(str) + "_x_" + df["PaymentMethod"].astype(str)
        df["cross_service_contract"] = df["service_combo_id"].astype(str) + "_x_" + df["Contract"].astype(str)
        df["cross_geo_contract"] = df["geo_code"].astype(str) + "_x_" + df["Contract"].astype(str)

        num_features = ["tenure", "MonthlyCharges"]
        cat_features = ["Contract", "PaymentMethod", "gender", "cross_contract_payment", "cross_service_contract", "cross_geo_contract"]
        hash_features = ["service_combo_id", "geo_code"]

        X = df[num_features + cat_features + hash_features]
        y = df['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("hash", HashingTransformer(cols=hash_features), hash_features)
        ])

        pipeline = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=42)),
            ("model", RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42))
        ])

        print("🏗️ Model eğitiliyor...")
        pipeline.fit(X_train, y_train)

        print("💾 Kayıt başlıyor...")
        os.makedirs(self.project_root / "models", exist_ok=True)
        joblib.dump(pipeline, self.project_root / "models" / "model.joblib")
        print("✅ BAŞARILI! model.joblib dosyası güncellendi.")

if __name__ == "__main__":
    trainer = CleanedDataTrainer()
    trainer.run()