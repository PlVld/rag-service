# =============================================================================
# Этап 0: Сборщик зависимостей
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

# ---------------------------------------------------------------------------
# Непривилегированный пользователь (до COPY --chown)
# ---------------------------------------------------------------------------
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Копирование установленных пакетов из builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Модели Docling. Путь /app/models ищет app/text_cleaning/bundled_tools.py
COPY --from=docling-models --chown=appuser:appuser /docling_models /app/models

WORKDIR /app

# Эмбеддинги считаются в отдельном контейнере (text-embeddings-inference),
# поэтому модель bge-m3 в образ не встраивается (EMBEDDINGS_API_URL в compose).
# HF_HUB_OFFLINE=1 — docling-модели загружаются из /app/models, сети не нужно.
ENV HF_HUB_OFFLINE=1

# Переменная окружения для poppler (Linux)
ENV POPPLER_PATH=/usr/bin

# Копирование исходного кода
COPY --chown=appuser:appuser ./app /app/app

# Каталог для загружаемых файлов (volume mount из docker-compose)
RUN mkdir -p /app/uploads && chown -R appuser:appuser /app/uploads

# Порт сервиса (можно переопределить через docker-compose или .env)
ENV SERVICE_PORT=8000
ENV SERVICE_HOST=0.0.0.0

# Открытие порта
EXPOSE 8000

USER appuser

# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Запуск приложения
CMD ["sh", "-c", "uvicorn app.main:app --host $SERVICE_HOST --port $SERVICE_PORT"]
