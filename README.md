# 📄 Universal Document Vector Search Service

> **Микросервис для семантического поиска по документам с поддержкой категорий, версионирования и MCP-интеграцией**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135%2B-009688)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-FF6F61)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

[![GitHub Stars](https://img.shields.io/github/stars?style=social)](https://github.com/USERNAME/rag-service/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/USERNAME/rag-service)](https://github.com/USERNAME/rag-service/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/USERNAME/rag-service)](https://github.com/USERNAME/rag-service/discussions)

### ✨ Ключевые возможности

- 🔍 **Семантический поиск** — поиск по смыслу текста, а не по ключевым словам
- 📊 **Гибридный поиск** — RRF (Reciprocal Rank Fusion) для лучшего качества
- 📁 **Много форматов** — PDF, DOCX, TXT, HTML, Markdown, XLSX
- 🗂️ **Категоризация** — автоматическая иерархическая система категорий
- 🔄 **Версионирование** — отслеживание изменений документов
- 🤖 **MCP Integration** — готовая интеграция с AI-ассистентами
- 🐳 **Docker Ready** — быстрое развёртывание за 2 минуты
- 🎨 **GPU Acceleration** — ускорение через NVIDIA CUDA

## 📝 О проекте

Этот сервис позволяет загружать документы различных форматов, автоматически извлекать из них текст, разбивать на чанки, создавать векторные представления и выполнять семантический поиск. Результаты можно группировать по категориям и фильтраровать по метаданным.

Проект идеален для:
- **B2B интеграций** — поиск по технической документации, инструкциям, базам знаний
- **AI-ассистентов** — MCP-протокол для подключения к LLM-ботам
- **Аналитики** — категоризация и поиск по большим массивам документов

### 📸 Скриншоты

> _Swagger UI документация_ (добавьте скриншот после запуска: `http://localhost:8000/docs`)

---

## 📋 Содержание

- [Возможности](#возможности)
- [Архитектура](#архитектура)
- [Требования](#требования)
- [Быстрый старт](#быстрый-старт)
- [Установка и настройка](#установка-и-настройка)
- [REST API](#rest-api)
- [MCP Интеграция](#mcp-интеграция)
- [Конфигурация](#конфигурация)
- [Развёртывание](#развёртывание)
- [Структура проекта](#структура-проекта)
- [Разработка](#разработка)
- [Тестирование](#тестирование)
- [FAQ](#faq)

## ✨ Возможности

- **Семантический поиск** по документам с использованием векторных эмбеддингов
- **Гибридный поиск** сweighted RRF (Reciprocal Rank Fusion)
- **Группировка результатов** по категориям и коллекциям
- **Поддержка множества форматов**: PDF, DOCX, DOC, TXT, HTML, Markdown, XLSX
- **OCR для сканированных документов** (Tesseract)
- **Конвертация документов в Markdown** через Docling
- **Иерархическая система категорий** с несколькими уровнями вложенности
- **Версионирование документов** с возможностью доступа к старым версиям
- **MCP (Model Context Protocol)** интеграция для AI-ассистентов
- **Docker-развёртывание** с поддержкой GPU
- **Аутентификация** для API и MCP endpoints

## 🏗️ Архитектура

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────┐
│   AI Client     │◄────►│  RAG Service     │◄────►│   Qdrant     │
│  (MCP/API)      │      │  (FastAPI)       │      │ (Vector DB)  │
└─────────────────┘      └──────────────────┘      └──────────────┘
                                  │
                                  ▼
                          ┌──────────────────┐
                          │  Document Store  │
                          │   (uploads/)     │
                          └──────────────────┘
```

**Основные компоненты:**
- **FastAPI** — веб-фреймворк для REST API и MCP прокси
- **Qdrant** — векторная база данных для хранения и поиска эмбеддингов
- **Sentence Transformers** — генерация векторных представлений текста
- **Docling** — конвертация документов в структурированный Markdown
- **Tesseract OCR** — распознавание текста на изображениях

## 📦 Требования

### Для локальной разработки:
- Python 3.11+
- Qdrant (локально или Docker)
- ~2GB свободного места для моделей эмбеддингов

### Для production (Docker):
- Docker Engine 20.10+
- Docker Compose v2
- GPU (опционально, для ускорения работы с эмбеддингами)

## ⚡ Быстрый старт (2 минуты)

### Запуск одним командой

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/YOUR_USERNAME/rag-service.git
cd rag-service

# 2. Настройте переменные окружения
cp .env.example .env
# Отредактируйте .env: укажите свой RAG_SERVICE_API_KEY

# 3. Запустите
docker-compose up -d

# 4. Готово! Откройте:
curl http://localhost:8000/health
# http://localhost:8000/docs (Swagger UI)
```

[Подробная инструкция →](#установка-и-настройка)

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd rag
```

### 2. Настройка переменных окружения

```bash
cp .env.example .env
```

Отредактируйте `.env` файл, указав необходимые значения:

```env
RAG_SERVICE_API_KEY=your-secret-api-key-here
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=BAAI/bge-m3
USE_GPU=false
```

### 3. Запуск с Docker Compose

```bash
docker-compose up -d
```

Это запустит:
- **Qdrant** на порту `6333`
- **RAG Service** на порту `8000`

### 4. Проверка работоспособности

```bash
# Проверка health check
curl http://localhost:8000/health

# Открытие Swagger UI
# http://localhost:8000/docs
```

## 📖 Установка и настройка

### Локальная установка (без Docker)

#### 1. Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Установка зависимостей
pip install -r requirements.txt
```

#### 2. Запуск Qdrant

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v ./qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

#### 3. Запуск приложения

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Использование CPU-only образа

```bash
docker-compose -f docker-compose.yml -f Dockerfile.cpu up -d
```

## 🌐 REST API

### Основные эндпоинты

#### Документы

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/documents/upload` | Загрузка документа |
| POST | `/api/documents/search` | Поиск документов |
| POST | `/api/documents/search/grouped` | Группированный поиск |

#### Категории

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/categories/` | Список категорий |
| POST | `/api/categories/search` | Поиск категорий |
| GET | `/api/categories/hierarchy` | Иерархия категорий |

#### Файлы

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/files/upload` | Загрузка одного файла |
| POST | `/api/files/upload/batch` | Пакетная загрузка файлов |

#### Администрирование

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/admin/collections` | Управление коллекциями |
| POST | `/api/admin/index` | Создание индексов |
| POST | `/api/admin/hnsw` | Настройка HNSW индекса |

### Интерактивная документация

После запуска откройте:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 MCP Интеграция

Сервис поддерживает **Model Context Protocol (MCP)** для интеграции с AI-ассистентами.

### Доступные инструменты

1. **`search_documents_tool`** — семантический поиск по документам
2. **`search_categories_tool`** — поиск категорий по запросу
3. **`get_category_hierarchy_tool`** — получение иерархии категорий

### Подключение

```
URL: http://localhost:8000/mcp
Method: POST
Content-Type: application/json
Authorization: Bearer <RAG_SERVICE_API_KEY>
```

### Пример вызова (tools/call)

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_documents_tool",
    "arguments": {
      "query_text": "настройка векторного поиска",
      "collection_name": "documents",
      "limit": 5
    }
  }
}
```

Подробная документация по MCP: [docs/mcp_tools.md](docs/mcp_tools.md)

## ⚙️ Конфигурация

Все настройки управляются через переменные окружения (файл `.env`).

### Ключевые параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `QDRANT_URL` | `http://localhost:6333` | URL Qdrant |
| `QDRANT_API_KEY` | — | API-ключ Qdrant |
| `RAG_SERVICE_API_KEY` | — | API-ключ сервиса (обязательный) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Модель эмбеддингов |
| `USE_GPU` | `false` | Использовать GPU |
| `SERVICE_PORT` | `8000` | Порт сервиса |
| `CHUNK_SIZE` | `1024` | Размер чанка (символы) |
| `CHUNK_OVERLAP` | `50` | Перекрытие чанков |
| `SEMANTIC_WEIGHT` | `0.3` | Вес семантического поиска |
| `CATEGORY_WEIGHT` | `0.7` | Вес категориального поиска |
| `ALLOWED_MCP_TOOLS` | — | Список разрешённых MCP инструментов |

Полный список переменных: [.env.example](.env.example)

## 🐳 Развёртывание

### Production deployment

1. Настройте `.env` для production:
   ```env
   USE_GPU=true
   LOG_LEVEL=WARNING
   MCP_AUTH_ENABLED=true
   ```

2. Соберите образ:
   ```bash
   docker build -t rag-service:latest -f Dockerfile .
   ```

3. Запустите:
   ```bash
   docker-compose up -d
   ```

### Horizontal Scaling

Для масштабирования можно запустить несколько инстансов сервиса за load balancer'ом, так как состояние хранится в Qdrant.

### Backup

Регулярно бэкапьте:
- Директорию `qdrant_data/` — данные векторной БД
- Директорию `uploads/` — исходные документы

## 📁 Структура проекта

```
rag/
├── app/                          # Основной код приложения
│   ├── api/                      # REST API endpoints
│   │   ├── documents.py          # Эндпоинты для документов
│   │   ├── categories.py         # Эндпоинты для категорий
│   │   ├── files.py              # Эндпоинты для файлов
│   │   ├── admin.py              # Админ-эндпоинты
│   │   ├── health.py             # Health check
│   │   └── ...
│   ├── core/                     # Ядро приложения
│   │   ├── config.py             # Конфигурация
│   │   └── embeddings.py         # Работа с эмбеддингами
│   ├── models/                   # Pydantic модели
│   ├── repository/               # Работа с Qdrant
│   ├── text_cleaning/            # Очистка и预处理 текста
│   │   ├── doc_cleaner.py        # Общий cleaner
│   │   ├── pdf_cleaner.py        # PDF специфичный
│   │   ├── markdown_cleaner.py   # Markdown специфичный
│   │   └── ...
│   ├── main.py                   # Точка входа (FastAPI app)
│   └── mcp_server.py             # MCP сервер
├── tests/                        # Тесты
├── docs/                         # Документация
│   ├── api/                      # Документация по API
│   ├── architecture.md           # Архитектура
│   ├── DEPLOYMENT.md             # Развёртывание
│   └── mcp_tools.md              # MCP инструменты
├── uploads/                      # Загруженные документы
├── qdrant_data/                  # Данные Qdrant
├── model_cache/                  # Кэш моделей
├── docker-compose.yml            # Docker Compose конфиг
├── Dockerfile                    # Docker образ (GPU)
├── Dockerfile.cpu                # Docker образ (CPU)
├── requirements.txt              # Зависимости Python
└── .env.example                  # Пример конфигурации
```

## 🧪 Тестирование

### Запуск всех тестов

```bash
pytest
```

### Запуск с покрытием

```bash
pytest --cov=app --cov-report=html
```

### Запуск конкретных тестов

```bash
# Тесты API документов
pytest tests/test_rest_api_documents.py -v

# Тесты категорий
pytest tests/test_rest_api_categories.py -v

# Тесты MCP сервера
pytest tests/test_mcp_server.py -v
```

## ❓ FAQ

### Как загрузить документы?

Используйте POST `/api/documents/upload` или через Swagger UI на `http://localhost:8000/docs`.

### Как искать документы?

POST `/api/documents/search` с телом запроса:
```json
{
  "query_text": "ваш запрос",
  "limit": 10
}
```

### Как использовать GPU?

1. Установите NVIDIA Docker runtime
2. В `docker-compose.yml` раскомментируйте секцию `deploy` с GPU
3. Установите `USE_GPU=true` в `.env`

### Как сменить модель эмбеддингов?

Укажите другую модель в `.env`:
```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Как настроить авторизацию?

Установите `RAG_SERVICE_API_KEY` в `.env` и передавайте заголовок:
```
Authorization: Bearer <your-api-key>
```

## 🛠️ Tech Stack

| Компонент | Технология |
|-----------|-----------|
| **Backend** | Python 3.11, FastAPI |
| **Vector DB** | Qdrant |
| **Embeddings** | Sentence Transformers (BAAI/bge-m3) |
| **Document Processing** | Docling, LangChain |
| **OCR** | Tesseract |
| **Deployment** | Docker, Docker Compose, Kubernetes |
| **API** | REST + MCP (Model Context Protocol) |

---

## 🎬 Quick Demo

```bash
# Загрузить документ
curl -X POST http://localhost:8000/v1/documents/upload \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"text": "Текст для поиска", "category_path": "Документация"}]}'

# Искать
curl -X POST http://localhost:8000/v1/documents/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "как искать документы", "limit": 5}'
```

---

## 📁 Related Projects

- [Qdrant](https://github.com/qdrant/qdrant) — Vector Database
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Docling](https://github.com/DS4SD/docling) — Document Conversion

---

## 📄 Лицензия

[MIT License](LICENSE)

---

## 🤝 Вклад в проект

Мы приветствуем contributions! Пожалуйста:

1. Fork репозитория
2. Создайте ветку (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

Подробности в [CONTRIBUTING.md](docs/CONTRIBUTING.md) (если есть)

---

## 👥 Авторы

Разработано командой в [Ваша компания/Org]

- [Your Name](https://github.com/YOUR_USERNAME) — Initial work

---

## 📬 Контакты

- **Email**: your@email.com
- **GitHub Issues**: [Открыть issue](https://github.com/YOUR_USERNAME/rag-service/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/rag-service/discussions)
