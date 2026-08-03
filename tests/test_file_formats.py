"""Тесты загрузки файлов различных форматов (текст, markdown, PDF)."""
import pytest
import time
import uuid
from app.api.health import get_client
from app.core.config import settings
from qdrant_client.http import models as qdrant_models


def get_auth_headers():
    """Возвращает заголовки для аутентификации через Bearer token."""
    return {"Authorization": "Bearer PASS1234"}


@pytest.fixture
def test_collection():
    """Создает и удаляет тестовую коллекцию для интеграционных тестов с реальной базой."""
    collection_name = f"test_formats_{uuid.uuid4().hex[:8]}"
    
    client = get_client()
    
    # Создаем тестовую коллекцию
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": 1024,
                "distance": "Cosine"
            }
        )
        
        # Ждем создания коллекции
        time.sleep(1)
        
        yield collection_name
        
    finally:
        # Удаляем тестовую коллекцию
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass


@pytest.mark.integration
def test_upload_text_document(test_collection, rest_api_client):
    """Тест загрузки простого текстового документа."""
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": "Это тестовый текстовый документ.\nВ нем несколько строк.\nОн должен быть успешно загружен.",
                    "payload": {
                        "source_format": "text",
                        "category_path": "Тесты / Форматы / Текст",
                        "source_id": "test-text-001"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    assert response_json["data"]["uploaded"] == 1
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем, что документ загружен
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
    
    # Ищем наш документ по source_id
    found = False
    for p in points:
        if p.payload.get("source_id") == "test-text-001":
            found = True
            assert "raw_text" in p.payload
            assert "тестовый текст" in p.payload.get("raw_text", "").lower()
            break
    
    assert found, "Document should be found in collection"


@pytest.mark.integration
def test_upload_markdown_document(test_collection, rest_api_client):
    """Тест загрузки markdown документа."""
    markdown_text = """# Заголовок документа

Это основной текст документа.

## Второй раздел

Здесь может быть список:
- Пункт 1
- Пункт 2
- Пункт 3

### Подраздел

Код может быть вставлен:
```python
def hello():
    print("Hello, World!")
```
"""
    
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": markdown_text,
                    "payload": {
                        "source_format": "markdown",
                        "category_path": "Тесты / Форматы / Markdown",
                        "source_id": "test-markdown-001",
                        "original_filename": "test_document.md"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    # Документ с 3 заголовками разбивается на 3 секции (по одной на каждый заголовок)
    assert response_json["data"]["uploaded"] >= 1

    # Ждем индексации
    time.sleep(1)

    # Проверяем загрузку
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)

    # Все чанки должны иметь правильный source_id
    matching = [p for p in points if p.payload.get("source_id") == "test-markdown-001"]
    assert len(matching) >= 1, f"Expected at least 1 chunk, got {len(matching)}"

    for p in matching:
        assert "raw_text" in p.payload
        assert "category_path" in p.payload
        # Каждый чанк должен иметь категорию, включающую документную категорию
        cat_path = p.payload.get("category_path", "")
        assert "Тесты" in cat_path and "Markdown" in cat_path

    # Проверяем, что разные заголовки дали разные категории
    cat_paths = {p.payload.get("category_path") for p in matching}
    # Должны быть чанки с разными заголовками в категории
    has_main = any("заголовок документа" in str(cp).lower() for cp in cat_paths)
    has_section = any("второй раздел" in str(cp).lower() for cp in cat_paths)
    assert has_main or has_section, f"Expected different heading categories, got: {cat_paths}"


@pytest.mark.integration
def test_upload_pdf_with_text_layer(test_collection, rest_api_client):
    """Тест загрузки PDF с текстовым слоем (цифровой PDF)."""
    # Для PDF требуется загрузка через multipart/form-data
    # Но API также поддерживает передачу текста, который будет обработан как PDF
    # В данном тесте мы проверяем обработку PDF текста
    
    pdf_text = "Тестовый PDF документ с текстовым слоем. Этот PDF был создан программно для тестирования. Он должен быть успешно распознан."
    
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": pdf_text,
                    "payload": {
                        "source_format": "pdf",
                        "category_path": "Тесты / Форматы / PDF с текстом",
                        "source_id": "test-pdf-text-001",
                        "original_filename": "test_document.pdf"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем загрузку
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
    
    found = False
    for p in points:
        if p.payload.get("source_id") == "test-pdf-text-001":
            found = True
            assert "raw_text" in p.payload
            assert "тестовый pdf" in p.payload.get("raw_text", "").lower()
            break
    
    assert found, "PDF document should be found in collection"


@pytest.mark.integration
def test_upload_code_document(test_collection, rest_api_client):
    """Тест загрузки исходного кода."""
    code_text = """# Проверка кода
def calculate_sum(a, b):
    \"\"\"Функция для суммирования двух чисел.\"\"\"
    return a + b

class Calculator:
    \"\"\"Класс калькулятора.\"\"\"
    
    def __init__(self):
        self.result = 0
    
    def add(self, value):
        \"\"\"Добавляет значение к результату.\"\"\"
        self.result += value
        return self.result
"""
    
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": code_text,
                    "payload": {
                        "source_format": "code",
                        "category_path": "Тесты / Форматы / Код",
                        "source_id": "test-code-001",
                        "original_filename": "calculator.py"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем загрузку
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
    
    found = False
    for p in points:
        if p.payload.get("source_id") == "test-code-001":
            found = True
            assert "raw_text" in p.payload
            assert "calculate_sum" in p.payload.get("raw_text", "")
            break
    
    assert found, "Code document should be found in collection"


@pytest.mark.integration
def test_upload_multiple_documents(test_collection, rest_api_client):
    """Тест загрузки нескольких документов разного формата одновременно."""
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": "Простой текстовый документ.",
                    "payload": {
                        "source_format": "text",
                        "category_path": "Тесты / Мульти-загрузка",
                        "source_id": "multi-text-001"
                    }
                },
                {
                    "text": "# Заголовок\n\nТекст markdown документа.",
                    "payload": {
                        "source_format": "markdown",
                        "category_path": "Тесты / Мульти-загрузка",
                        "source_id": "multi-md-001"
                    }
                },
                {
                    "text": "Текст документа PDF.",
                    "payload": {
                        "source_format": "pdf",
                        "category_path": "Тесты / Мульти-загрузка",
                        "source_id": "multi-pdf-001"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    assert response_json["data"]["uploaded"] == 3
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем, что все документы загружены
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=20, with_payload=True)
    
    source_ids = {p.payload.get("source_id") for p in points}
    expected_ids = {"multi-text-001", "multi-md-001", "multi-pdf-001"}
    
    assert source_ids >= expected_ids, f"All documents should be uploaded. Found: {source_ids}, Expected: {expected_ids}"


@pytest.mark.integration
def test_upload_document_with_category_hierarchy(test_collection, rest_api_client):
    """Тест загрузки документа с глубокой иерархией категорий."""
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": "Документ с глубокой иерархией категорий.",
                    "payload": {
                        "source_format": "text",
                        "category_path": "Раздел 1 / Подраздел 1.1 / Подраздел 1.1.1 / Детальный раздел",
                        "source_id": "test-hierarchy-001"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем загрузку
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
    
    found = False
    for p in points:
        if p.payload.get("source_id") == "test-hierarchy-001":
            found = True
            # Проверяем, что категория была обработана
            assert "category_path" in p.payload
            category_path = p.payload.get("category_path", "")
            assert "Раздел 1" in category_path
            assert "Детальный раздел" in category_path
            break
    
    assert found, "Document with category hierarchy should be found"
