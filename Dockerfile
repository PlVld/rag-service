# =============================================================================
# Этап 0: Скачивание модели bge-m3
# =============================================================================
FROM python:3.11-slim AS model-downloader

RUN pip install --no-cache-dir sentence-transformers transformers huggingface_hub
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-m3')"

# =============================================================================
# Этап 1: Сборщик зависимостей
# =============================================================================
FROM python:3.11-slim AS builder

# Установка системных зависимостей для компиляции
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Установка зависимостей из PyPI (кэшируется Docker BuildKit)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# =============================================================================
# Этап 2: Финальный образ
# =============================================================================
FROM python:3.11-slim

# Установка системных зависимостей для OCR и обработки документов
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

# Копирование установленных пакетов из builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копирование модели bge-m3
COPY --from=model-downloader /root/.cache/huggingface/hub /app/model_cache/hub

WORKDIR /app

# Создаём пустые директории для моделей
RUN mkdir -p /app/model_cache /app/docling_cache /app/uploads

# Копирование исходного кода
COPY ./app /app/app

# Переменные окружения для кэша моделей.
# Модели теперь встроены в Docker-образ, HF_HUB_OFFLINE=1 предотвращает попытки подключения к интернету.
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
ENV HF_HUB_OFFLINE=1

# Переменные для Docling моделей (встроены в образ)
ENV DOCLING_MODELS_PATH=/app/docling_cache

# Переменная окружения для poppler (Linux)
ENV POPPLER_PATH=/usr/bin

# Порт сервиса (можно переопределить через docker-compose или .env)
ENV SERVICE_PORT=8000
ENV SERVICE_HOST=0.0.0.0

# Открытие порта
EXPOSE 8000

# Запуск приложения
CMD ["sh", "-c", "uvicorn app.main:app --host $SERVICE_HOST --port $SERVICE_PORT"]
