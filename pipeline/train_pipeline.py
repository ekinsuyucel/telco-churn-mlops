import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Hoca Ensemble pattern istiyor 
from sklearn.metrics import average_precision_score
import joblib
import os
import mlflow # MLflow zorunlu [cite: 13, 15]

def main():
    # MLflow Deney Başlatma [cite: 16]
    mlflow.set_experiment("Telco_Churn_Production")
    
    with mlflow.start_run():
        print("🚀 Training pipeline started with High-Cardinality features")

        # 1. Veriyi Yükle (Kişi 3'ün temizlediği gerçek veri)
        # Veri yolu senin klasör yapına göre 'data/telco_cleaned.csv' olmalı
        df = pd.read_csv('data/telco_cleaned.csv')

        # 2. Zorunlu Özellikler (Hocanın istediği High-Cardinality kısımları) [cite: 31]
        # service_combo_id ve geo_code gibi alanları Kişi 1 ve 2 hazırladı.
        # Bunları modele sokmadan önce 'get_dummies' ile encode ediyoruz (Simple Hashing/Embedding muadili)
        features = ["tenure", "monthly_charges", "service_combo_id", "geo_code"]
        X = pd.get_dummies(df[features], columns=["service_combo_id", "geo_code"])
        y = df["churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # 3. Model Seçimi (Ensemble Pattern: RandomForest) 
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 4. Metrik Hesaplama (Hocanın istediği PR-AUC) [cite: 39, 47]
        preds = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, preds)

        # 5. MLflow Logging (ZORUNLU) [cite: 16]
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("pr_auc", pr_auc)
        
        # Model Registry'e kayıt (MLOps Level 2 gereği) [cite: 17]
        mlflow.sklearn.log_model(model, "model", registered_model_name="TelcoChurnModel")

        print(f"✅ PR-AUC: {pr_auc:.4f}")

        # 6. Artifact Kaydı (Senin API'nin okuyacağı yer)
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.joblib"
        joblib.dump(model, model_path)
    
        print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    main()