# Архитектура системы

## Обзор

Universal Document Vector Search Service — это микросервис для семантического поиска по документам, построенный на основе FastAPI и Qdrant. Система поддерживает загрузку документов различных форматов, их обработку, векторизацию и семантический поиск с возможностью группировки по категориям.

## High-Level Архитектура

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Клиенты                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │   Web    │  │   MCP    │  │  CLI     │  │  SDK     │           │
│  │  Client  │  │  Client  │  │  Tools   │  │  Libraries│           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │             │             │             │                  │
└───────┼─────────────┼─────────────┼─────────────┼──────────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG Service (FastAPI)                           │
│                     Port: 8000                                       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                    Middleware Layer                       │       │
│  │  Auth │ Diagnostics │ MCP Proxy │ Request Processing    │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  REST API   │  │   MCP API   │  │   Admin     │                 │
│  │  Endpoints  │  │  Endpoints  │  │  Endpoints  │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                   Business Logic Layer                    │       │
│  │  Documents  │  Categories  │  Search     │  Indexing    │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                Data Access Layer                          │       │
│  │  Qdrant Repository │ Embeddings │ Text Processing       │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────┐      ┌──────────────────┐      ┌──────────────┐
│   File System   │      │    Qdrant DB     │      │  Models Cache│
│                 │      │                  │      │              │
│  uploads/       │      │  • collections   │      │  • bge-m3    │
│  model_cache/   │      │  • categories    │      │  • docling   │
└─────────────────┘      └──────────────────┘      └──────────────┘
```

## Компоненты системы

### 1. API Layer (`app/api/`)

Отвечает за обработку HTTP-запросов и формирование ответов.

#### Компоненты:

- **documents.py** — загрузка и поиск документов
- **categories.py** — управление категориями
- **files.py** — работа с файлами (upload)
- **admin.py** — административные операции
- **health.py** — health check и мониторинг

#### Роутеры:

```
/v1/documents/upload     — POST: Загрузка документов
/v1/documents/search     — POST: Поиск документов
/v1/categories/          — GET: Получение категорий
/v1/categories/search    — POST: Поиск категорий
/api/files/upload        — POST: Загрузка файлов
/api/admin/*             — POST: Админ-операции
/health                  — GET: Health check
/mcp                     — POST: MCP endpoints
```

### 2. Core Layer (`app/core/`)

Базовая функциональность и конфигурация.

#### Компоненты:

- **config.py** — управление настройками через `.env`
- **embeddings.py** — генерация векторных представлений

#### Ключевые функции:

```python
async def encode_text(texts, task_type="query")
    # Генерирует эмбеддинги для списка текстов
    # Поддерживает batch processing
    # Кэширует результаты
```

### 3. Text Processing (`app/text_cleaning/`)

Обработка и очистка текста из различных форматов.

#### Pipeline обработки:

```
Raw Text → Format Detection → Cleaner → Normalizer → Markdown → Chunking
```

#### Компоненты:

| Модуль | Назначение |
|--------|-----------|
| `base.py` | Базовый класс для cleaner'ов |
| `doc_cleaner.py` | Общий cleaner для документов |
| `pdf_cleaner.py` | Обработка PDF |
| `markdown_cleaner.py` | Очистка Markdown |
| `html_cleaner.py` | Обработка HTML |
| `normalizer.py` | Нормализация текста для эмбеддингов |
| `pipeline.py` | Orchestration text cleaning pipeline |
| `heading_splitter.py` | Разбиение по заголовкам |
| `russian_cleaner.py` | Русскоязычная специфика |

#### Docling Integration:

```python
# Docling используется для конвертации сложных документов
# Поддерживает: PDF, DOCX, HTML, изображения
# Включает OCR для сканированных документов
```

### 4. Chunking (`app/chunking.py`)

Разбиение текста на чанки для векторизации.

#### Стратегии:

- **LangChain-based** — использование langchain text splitters
- **Custom** — собственная реализация с контролем overlap

#### Параметры:

```python
chunk_size=1024    # Размер чанка в символах
chunk_overlap=50   # Перекрытие между чанками
```

### 5. Repository Layer (`app/repository/`)

Абстракция над Qdrant API.

#### QdrantBatchWriter:

```python
class QdrantBatchWriter:
    # Буферизация точек для batch upsert
    # Оптимизация производительности
    # Transaction management
```

### 6. MCP Server (`app/mcp_server.py`)

Model Context Protocol интеграция.

#### Инструменты:

1. **search_documents_tool** — поиск по документам
2. **search_categories_tool** — поиск категорий
3. **get_category_hierarchy_tool** — иерархия категорий

#### Протокол:

```
Client → POST /mcp → JSON-RPC 2.0 → Tool Execution → Response
```

### 7. Models (`app/models/`)

Pydantic модели для валидации данных.

#### Ключевые модели:

```python
class DocumentSearchRequest(BaseModel):
    query_text: str
    collection_name: Optional[str]
    limit: int
    filter: Optional[dict]
    group: bool

class DocumentsUploadRequest(BaseModel):
    collection_name: str
    documents: List[DocumentCreate]

class CategorySearchRequest(BaseModel):
    query_text: str
    limit: int
    grouped: bool
```

## Бизнес-логика

### Процесс загрузки документов

```
1. Receive Upload Request
   ↓
2. Validate & Parse Documents
   ↓
3. Detect Format (PDF, DOCX, MD, etc.)
   ↓
4. Text Extraction (Docling / Legacy)
   ↓
5. Text Cleaning & Normalization
   ↓
6. Heading-based Section Splitting
   ↓
7. Category Path Construction
   ├─ Document categories
   └─ Heading hierarchy
   ↓
8. Chunking
   ├─ Apply chunk_size & chunk_overlap
   └─ Preserve metadata
   ↓
9. Embedding Generation
   └─ Batch encoding
   ↓
10. Category Hierarchy Update
    └─ Upsert category nodes
    ↓
11. Document Indexing (Qdrant)
    ├─ Generate point IDs
    ├─ Set is_latest flag
    └─ Batch upsert
    ↓
12. Response
```

### Процесс поиска

```
1. Receive Search Request
   ↓
2. Normalize Query Text
   ↓
3. Generate Query Embedding
   ↓
4. Collection Resolution
   ├─ Single collection
   └─ All collections (cross-collection search)
   ↓
5. Prefetch Queries
   ├─ Vector similarity
   └─ Category-based (if categories specified)
   ↓
6. RRF Fusion
   └─ Reciprocal Rank Fusion
   ↓
7. Reranking
   └─ Cosine similarity with query vector
   ↓
8. Grouping (optional)
   ├─ Group by category_path
   └─ Aggregate chunks
   ↓
9. Apply Filters & Limits
   ↓
10. Format & Return Results
```

### Гибридный поиск (RRF)

Система использует **Reciprocal Rank Fusion** для объединения результатов:

```python
# RRF формула
score = sum(1 / (k + rank))

# k — константа (обычно 60)
# rank — позиция элемента в ранжированном списке
```

#### Weighted RRF:

```python
# Итоговый score = doc_score * document_weight + category_score * category_weight_final
# Настройки в .env:
# DOCUMENT_WEIGHT=0.4
# CATEGORY_WEIGHT_FINAL=0.6
```

## Хранение данных

### Qdrant Collections

| Collection | Назначение |
|------------|-----------|
| `documents` (настраиваемое) | Основные документы и чанки |
| `categories` | Иерархия категорий |

### Структура точки документа:

```json
{
  "id": "uuid-source-version-chunk",
  "vector": [0.123, -0.456, ...],  // embedding vector
  "payload": {
    "source_id": "unique-source-id",
    "version": 1,
    "is_latest": true,
    "chunk_index": 0,
    "total_chunks": 5,
    "raw_text": "Original text...",
    "normalized_text": "Cleaned text...",
    "category_path": "Category / Subcategory",
    "category_level": 1,
    "category_id_level0": "uuid-cat-0",
    "category_id_level1": "uuid-cat-1",
    "original_filename": "example.pdf",
    "title": "Document Title",
    "source_format": "markdown",
    "content_type": "markdown",
    "doc_hash": "sha256-hash",
    "doc_category_count": 2
  }
}
```

### Структура точки категории:

```json
{
  "id": "uuid-category-path",
  "vector": [0.789, 0.321, ...],
  "payload": {
    "category_name": "Subcategory",
    "category_path": "Category / Subcategory",
    "category_level": 1,
    "parent_id": "uuid-parent-category",
    "category_id": "uuid-this-category",
    "category_id_level0": "uuid-cat-0",
    "category_id_level1": "uuid-cat-1"
  }
}
```

## Конфигурация

### Переменные окружения

Ключевые настройки в `.env`:

```env
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-m3
USE_GPU=false

# Service
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
RAG_SERVICE_API_KEY=your-key

# Chunking
CHUNK_SIZE=1024
CHUNK_OVERLAP=50

# Search
SEMANTIC_WEIGHT=0.3
CATEGORY_WEIGHT=0.7
ENABLE_WEIGHTED_RRF=true

# Docling
USE_DOCLING=true
DOCLING_OCR_ENGINE=tesseract
```

## Масштабируемость

### Vertical Scaling

- Увеличение RAM для обработки больших батчей
- GPU для ускорения эмбеддингов

### Horizontal Scaling

```
[Load Balancer]
    │
    ├─ [RAG Service Instance 1] ──┐
    ├─ [RAG Service Instance 2] ──┼─→ [Qdrant Cluster]
    └─ [RAG Service Instance N] ──┘
```

- Stateless service — можно запускать множественные инстансы
- Общее состояние в Qdrant
- Кэширование на уровне приложения (TTL 60s)

## Безопасность

### Аутентификация

```python
# API Key authentication
Authorization: Bearer <RAG_SERVICE_API_KEY>

# MCP authentication (опционально)
MCP_AUTH_ENABLED=true
```

### Авторизация MCP

```env
# Разрешить только определённые инструменты
ALLOWED_MCP_TOOLS=search_documents_tool,search_categories_tool
```

### Валидация данных

- Pydantic модели для всех входных данных
- Валидация файлов (размер, MIME-тип)
- Санитизация текста

## Monitoring & Logging

### Middleware

1. **RequestDiagnosticsMiddleware** — логирование всех запросов
2. **ConsumeRequestBodyMiddleware** — обработка chunked encoding
3. **MCP Logging Middleware** — детальное логирование MCP

### Логи

```python
# Уровни логирования
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Специфичные логи
diag_logger  — diagnostics
logger       — стандартное логирование
```

## Зависимости

### Основные

- **FastAPI** — веб-фреймворк
- **Qdrant Client** — работа с векторной БД
- **Sentence Transformers** — эмбеддинги
- **LangChain** — text splitters

### Обработка документов

- **Docling** — конвертация PDF/DOCX/HTML
- **python-docx** — работа с DOCX
- **pypdf** — работа с PDF
- **BeautifulSoup4** — парсинг HTML

### OCR

- **Tesseract** — OCR движок
- **pdf2image** — конвертация PDF → images

## Развёртывание

### Docker Compose

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
  
  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
```

### Production

См. [DEPLOYMENT.md](DEPLOYMENT.md)

## Разработка

### Структура тестов

```
tests/
├── conftest.py                    # Fixtures
├── test_rest_api_documents.py     # Тесты документов
├── test_rest_api_categories.py    # Тесты категорий
├── test_rest_api_integration.py   # Integration tests
└── test_mcp_server.py             # Тесты MCP
```

### Запуск

```bash
# Development
uvicorn app.main:app --reload

# Tests
pytest
```

## Версионирование

- API версия: `v1` (в path: `/v1/documents`)
- Версия сервиса: `1.2.0` (в FastAPI metadata)
- Версия MCP: `2024-11-05`
