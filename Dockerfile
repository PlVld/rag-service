# =============================================================================n# Этап 0: Загрузка модели bge-m3
# =============================================================================
FROM python:3.11-slim AS model-downloader

RUN pip install --no-cache-dir sentence-transformers transformers huggingface_hub
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-m3', local_files_only=False)"

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
# Этап 1.5: Модели Docling (layout + tableformer)
# Кладём в образ, потому что в финальной стадии стоит HF_HUB_OFFLINE=1
# и скачать их в рантайме уже нельзя.
# =============================================================================
FROM builder AS docling-models

RUN docling-tools models download -o /docling_models layout tableformer

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

# Копирование модели bge-m3 из model-downloader
RUN mkdir -p /app/model_cache
COPY --from=model-downloader /root/.cache/huggingface/hub /app/model_cache/hub

# Модели Docling. Путь /app/models ищет app/text_cleaning/bundled_tools.py
COPY --from=docling-models /docling_models /app/models

WORKDIR /app

# Переменные окружения для кэша моделей.
# Модель mount'ится из ./model_cache через docker-compose (volume mount)
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
ENV HF_HUB_OFFLINE=1

# Переменная окружения для poppler (Linux)
ENV POPPLER_PATH=/usr/bin

# Копирование исходного кода
COPY ./app /app/app

# Порт сервиса (можно переопределить через docker-compose или .env)
ENV SERVICE_PORT=8000
ENV SERVICE_HOST=0.0.0.0

# Открытие порта
EXPOSE 8000

# ---------------------------------------------------------------------------
# Непривилегированный пользователь
# ---------------------------------------------------------------------------
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
RUN chown -R appuser:appuser /app
USER appuser

# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Запуск приложения
CMD ["sh", "-c", "uvicorn app.main:app --host $SERVICE_HOST --port $SERVICE_PORT"]
