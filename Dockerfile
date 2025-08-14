FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Установка системных пакетов, включая redis-cli
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

CMD ["python", "main.py"]