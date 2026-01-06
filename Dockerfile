FROM python:3.11-slim

WORKDIR /app

# Bağımlılıkları kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları ve modeli kopyala
COPY app/ ./app/
COPY models/ ./models/

# API'yi çalıştır
CMD ["uvicorn", "app.main:app", "--host", "0.0.1", "--port", "8000"]