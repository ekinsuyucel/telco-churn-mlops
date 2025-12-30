# 1️⃣ Base image
FROM python:3.10-slim

# 2️⃣ Çalışma dizini
WORKDIR /app

# 3️⃣ Sistem bağımlılıkları (xgboost, numpy vs için)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Proje dosyaları
COPY scripts/ scripts/
COPY config/ config/
COPY data/ data/

# 6️⃣ Default çalışacak komut
CMD ["python", "scripts/train_cleaned_data.py", "--config", "config/telco_cleaned.yaml"]
