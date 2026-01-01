import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Ensemble pattern (hocanın istediği)
from sklearn.metrics import average_precision_score
import joblib
import os
import mlflow
import mlflow.sklearn


def main():
    # 1️⃣ MLflow experiment
    mlflow.set_experiment("Telco_Churn_Production")

    with mlflow.start_run():
        print("🚀 Training pipeline started with High-Cardinality features")

        # 2️⃣ Load cleaned data (Kişi 3)
        df = pd.read_csv("data/telco_cleaned.csv")

        # 🔒 Schema safety (gerçek MLOps refleksi)
        df.columns = [col.strip() for col in df.columns]

        print("📊 Available columns:", df.columns.tolist())

        # 3️⃣ Feature selection (DATA CANONICAL)
        features = [
            "tenure",
            "MonthlyCharges",
            "service_combo_id",
            "geo_code"
        ]

        # Target column (dataset’te 'Churn' var)
        target = "Churn"

        # 4️⃣ One-hot encoding (high-cardinality dahil)
        X = pd.get_dummies(
            df[features],
            columns=["service_combo_id", "geo_code"],
            drop_first=False
        )

        y = df[target].map({"Yes": 1, "No": 0})

        # 5️⃣ Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # 6️⃣ Ensemble model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # 7️⃣ Metric: PR-AUC (PRIMARY)
        preds = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, preds)

        # 8️⃣ MLflow logging (ZORUNLU)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("pr_auc", pr_auc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="TelcoChurnModel"
        )

        print(f"✅ PR-AUC: {pr_auc:.4f}")

        # 9️⃣ Local artifact (API için)
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.joblib"
        joblib.dump(model, model_path)

        print(f"✅ Model saved to {model_path}")


if __name__ == "__main__":
    main()
