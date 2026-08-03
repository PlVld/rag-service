# REST API Документация

## Базовая информация

- **База URL**: `http://localhost:8000`
- **Версия API**: v1
- **Аутентификация**: Bearer Token (`RAG_SERVICE_API_KEY`)
- **Content-Type**: `application/json`

## Аутентификация

Все эндпоинты требуют аутентификации. Передавайте API-ключ в заголовке:

```
Authorization: Bearer <your-api-key>
```

## Содержание

- [Documents API](#documents-api)
  - [Загрузка документов](#загрузка-документов)
  - [Поиск документов](#поиск-документов)
  - [Группированный поиск](#группированный-поиск)
- [Categories API](#categories-api)
  - [Поиск категорий](#поиск-категорий)
  - [Получение всех категорий](#получение-всех-категорий)
- [Files API](#files-api)
  - [Загрузка одного файла](#загрузка-одного-файла)
  - [Пакетная загрузка файлов](#пакетная-загрузка-файлов)
- [Admin API](#admin-api)
  - [Управление коллекциями](#управление-коллекциями)
  - [Создание индексов](#создание-индексов)
  - [Настройка HNSW](#настройка-hnsw)
- [Health Check](#health-check)
- [Форматы ответов](#форматы-ответов)
- [Коды ошибок](#коды-ошибок)

---

## Documents API

Префикс: `/v1/documents`

### Загрузка документов

**POST** `/v1/documents/upload`

Загружает документы в указанную коллекцию Qdrant.

#### Параметры

Можно отправить как `multipart/form-data`, так и JSON body.

| Параметр | Тип | Обязательный | Описание |
|----------|-----|-------------|----------|
| `documents` | array | ✅ | Массив объектов документов |
| `collection_name` | string | ❌ | Имя коллекции (по умолчанию из `.env`) |

#### Структура документа

```json
{
  "text": "Текст документа",
  "filename": "example.pdf",
  "title": "Название документа",
  "version": 1,
  "category_path": "Категория / Подкатегория",
  "payload": {
    "source_id": "optional-custom-id",
    "source_format": "markdown"
  }
}
```

#### Пример запроса

```bash
curl -X POST http://localhost:8000/v1/documents/upload \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents",
    "documents": [
      {
        "text": "# Пример документа\n\nТекст для индексации.",
        "filename": "example.md",
        "category_path": "Документация / Примеры"
      }
    ]
  }'
```

#### Пример ответа

```json
{
  "success": true,
  "data": {
    "uploaded": 5,
    "ids": ["uuid-point-1", "uuid-point-2", ...],
    "skipped": [
      {
        "source_id": "existing-doc",
        "reason": "already_up_to_date"
      }
    ]
  }
}
```

---

### Поиск документов

**POST** `/v1/documents/search`

Выполняет семантический поиск по документам.

#### Параметры

| Параметр | Тип | Обязательный | По умолчанию | Описание |
|----------|-----|-------------|-------------|----------|
| `query_text` | string | ✅ | — | Текст запроса |
| `collection_name` | string | ❌ | — | Имя коллекции (если не указано, ищет по всем) |
| `limit` | integer | ❌ | 10 | Количество результатов (1-1000) |
| `filter` | object | ❌ | — | Фильтр Qdrant |
| `include_old_versions` | boolean | ❌ | false | Включать старые версии документов |
| `max_text_length` | integer | ❌ | 0 | Максимальная длина текста (0 = без ограничений) |
| `group` | boolean | ❌ | true | Группировать по category_path |
| `payload_fields` | array | ❌ | — | Список полей для возврата |

#### Пример запроса

```bash
curl -X POST http://localhost:8000/v1/documents/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "настройка векторного поиска",
    "collection_name": "documents",
    "limit": 5,
    "group": true,
    "max_text_length": 1000
  }'
```

#### Пример ответа (с группировкой)

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "category_path": "Документация / API / Поиск",
        "score": 0.8542,
        "document": "Семантический поиск использует векторные представления...",
        "collection_name": "documents",
        "payload": {
          "filename": "search_guide.md",
          "version": 1
        }
      }
    ]
  }
}
```

---

### Группированный поиск

**POST** `/api/documents/search/grouped`

Специализированный поиск с взвешенным объединением score документов и категорий.

#### Параметры

| Параметр | Тип | Обязательный | По умолчанию | Описание |
|----------|-----|-------------|-------------|----------|
| `query_text` | string | ✅ | — | Текст запроса |
| `collection_name` | string | ❌ | — | Имя коллекции |
| `limit` | integer | ❌ | 10 | Количество результатов |
| `filter` | object | ❌ | — | Фильтр Qdrant |

#### Пример запроса

```bash
curl -X POST http://localhost:8000/api/documents/search/grouped \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "документация по интеграции",
    "collection_name": "documents",
    "limit": 10
  }'
```

---

## Categories API

Префикс: `/v1/categories`

### Поиск категорий

**POST** `/v1/categories/search`

Выполняет семантический поиск по категориям документов.

#### Параметры

| Параметр | Тип | Обязательный | По умолчанию | Описание |
|----------|-----|-------------|-------------|----------|
| `query_text` | string | ✅ | — | Текст запроса |
| `limit` | integer | ❌ | 10 | Количество результатов (1-100) |
| `grouped` | boolean | ❌ | false | Группировать по коллекциям |
| `fields` | array | ❌ | — | Поля для возврата |

#### Пример запроса

```bash
curl -X POST http://localhost:8000/v1/categories/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "API документация",
    "limit": 5,
    "grouped": true
  }'
```

#### Пример ответа (с группировкой)

```json
{
  "results": {
    "documents": [
      {
        "path": "Документация / API / Руководство",
        "score": 0.823
      }
    ],
    "not_found_in_documents": [
      {
        "path": "API / Спецификации",
        "score": 0.745
      }
    ]
  }
}
```

---

### Получение всех категорий

**GET** `/v1/categories/`

Возвращает список всех категорий.

#### Параметры

| Параметр | Тип | Обязательный | По умолчанию | Описание |
|----------|-----|-------------|-------------|----------|
| `limit` | integer | ❌ | 100 | Максимум результатов (1-1000) |
| `parent_id` | string | ❌ | — | Фильтр по родительской категории |

#### Пример запроса

```bash
curl -X GET "http://localhost:8000/v1/categories/?limit=50" \
  -H "Authorization: Bearer your-api-key"
```

#### Пример ответа

```json
{
  "results": [
    {
      "id": "uuid-1",
      "score": 1.0,
      "category_name": "Документация",
      "category_path": "Документация",
      "categories": ["Документация"],
      "category_level": 0,
      "category_id": "uuid-cat-1",
      "levels": {
        "category_level0": "Документация"
      },
      "id_levels": {
        "category_id_level0": "uuid-cat-1"
      }
    }
  ]
}
```

---

## Files API

Префикс: `/api/files`

### Загрузка одного файла

**POST** `/api/files/upload`

Загружает файл и автоматически извлекает текст.

#### Параметры

| Параметр | Тип | Обязательный | Описание |
|----------|-----|-------------|----------|
| `file` | file | ✅ | Файл для загрузки |
| `collection_name` | string | ❌ | Имя коллекции |
| `category_path` | string | ❌ | Путь категории |

#### Пример запроса

```bash
curl -X POST http://localhost:8000/api/files/upload \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@/path/to/document.pdf" \
  -F "collection_name=documents" \
  -F "category_path=Документация / PDF"
```

---

### Пакетная загрузка файлов

**POST** `/api/files/upload/batch`

Загружает несколько файлов одновременно.

#### Параметры

| Параметр | Тип | Обязательный | Описание |
|----------|-----|-------------|----------|
| `files` | array | ✅ | Массив файлов |
| `collection_name` | string | ❌ | Имя коллекции |

#### Пример запроса

```bash
curl -X POST http://localhost:8000/api/files/upload/batch \
  -H "Authorization: Bearer your-api-key" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.docx" \
  -F "files=@doc3.md"
```

---

## Admin API

Префикс: `/api/admin`

### Управление коллекциями

**POST** `/api/admin/collections`

Создаёт, удаляет или управляет коллекциями в Qdrant.

#### Пример запроса

```bash
curl -X POST http://localhost:8000/api/admin/collections \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "create",
    "collection_name": "new_collection"
  }'
```

---

### Создание индексов

**POST** `/api/admin/index`

Создаёт payload индексы для оптимизации поиска.

#### Пример запроса

```bash
curl -X POST http://localhost:8000/api/admin/index \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents",
    "field_name": "category_path",
    "field_type": "keyword"
  }'
```

---

### Настройка HNSW

**POST** `/api/admin/hnsw`

Настраивает параметры HNSW индекса.

#### Пример запроса

```bash
curl -X POST http://localhost:8000/api/admin/hnsw \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents",
    "m": 16,
    "ef_construct": 100
  }'
```

---

## Health Check

**GET** `/health`

Проверяет состояние сервиса и подключение к Qdrant.

#### Пример запроса

```bash
curl http://localhost:8000/health
```

#### Пример ответа

```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "timestamp": "2026-08-03T12:00:00Z"
}
```

---

## Форматы ответов

Все ответы имеют стандартную структуру:

### Успешный ответ

```json
{
  "success": true,
  "data": {
    // данные ответа
  }
}
```

### Ответ с ошибкой

```json
{
  "success": false,
  "error_code": "error_code_here",
  "error_message": "Описание ошибки"
}
```

---

## Коды ошибок

| HTTP Status | Код | Описание |
|-------------|-----|----------|
| 400 | `validation_error` | Ошибка валидации запроса |
| 401 | `unauthorized` | Неверный или отсутствующий API-ключ |
| 403 | `forbidden` | Недостаточно прав |
| 404 | `collection_not_found` | Коллекция не найдена |
| 413 | `file_too_large` | Файл слишком большой |
| 415 | `unsupported_media_type` | Неподдерживаемый тип файла |
| 500 | `internal_error` | Внутренняя ошибка сервера |
| 503 | `qdrant_unavailable` | Qdrant недоступен |

---

## SDK и клиенты

### Python пример

```python
import requests

API_KEY = "your-api-key"
BASE_URL = "http://localhost:8000"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Поиск документов
response = requests.post(
    f"{BASE_URL}/v1/documents/search",
    headers=headers,
    json={
        "query_text": "векторный поиск",
        "limit": 5
    }
)

results = response.json()["data"]["results"]
for result in results:
    print(f"[{result['score']:.3f}] {result['category_path']}")
    print(result['document'][:200])
```

### cURL примеры

Все примеры запросов приведены в соответствующих разделах выше.

---

## Интерактивная документация

После запуска сервиса доступна интерактивная документация:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
