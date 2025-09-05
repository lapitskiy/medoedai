FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем приложение
COPY . /app

ENV PYTHONUNBUFFERED=1 \
    MODEL_BASE_DIR=/workspace/models \
    PYTHONPATH=/workspace:/app

EXPOSE 8000