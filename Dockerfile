FROM python:3.11-slim

WORKDIR /app

# Bağımlılıkları kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını ve modeli kopyala
COPY app/ ./app/
COPY models/ ./models/

# API'yi python modülü üzerinden çalıştır (En güvenli yol)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]