"""Интеграционные тесты REST API endpoints с реальной базой данных.

Эти тесты используют реальную базу данных Qdrant для проверки полного цикла:
1. Создание тестовой коллекции
2. Загрузка и поиск документов
3. Поиск категорий
4. Удаление тестовой коллекции после завершения
"""
import pytest
import time
import uuid
from app.api.health import get_client


def get_auth_headers():
    """Возвращает заголовки для аутентификации через Bearer token."""
    return {"Authorization": "Bearer PASS1234"}


@pytest.fixture
def test_collection():
    """Создает и удаляет тестовую коллекцию для интеграционных тестов с реальной базой."""
    collection_name = f"test_rest_api_{uuid.uuid4().hex[:8]}"
    
    client = get_client()
    
    # Создаем тестовую коллекцию
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": 1024,  # BAAI/bge-m3 использует 1024 размерность
                "distance": "Cosine"
            }
        )
        
        # Заполняем коллекцию тестовыми данными
        from qdrant_client.http import models as qdrant_models
        
        # Генерируем тестовые эмбеддинги (1024-мерные векторы для BAAI/bge-m3)
        import random
        random.seed(42)  # Для воспроизводимости
        
        points = [
            qdrant_models.PointStruct(
                id=uuid.uuid4().hex,
                vector=[random.gauss(0, 1) for _ in range(1024)],
                payload={
                    "raw_text": "Тестовый документ для интеграционного теста REST API",
                    "source_id": "test-rest-doc-1",
                    "source_format": "text",
                    "is_latest": True,
                    "category_path": "Документация / REST API",
                    "original_filename": "test_rest_doc.md",
                    "content_type": "markdown",
                    "version": 1,
                    "chunk_index": 0,
                    "total_chunks": 3,
                }
            ),
            qdrant_models.PointStruct(
                id=uuid.uuid4().hex,
                vector=[random.gauss(0, 1) for _ in range(1024)],
                payload={
                    "raw_text": "Пример использования REST API для разработчиков",
                    "source_id": "test-rest-doc-2",
                    "source_format": "text",
                    "is_latest": True,
                    "category_path": "Документация / Примеры",
                    "original_filename": "test_rest_examples.md",
                    "content_type": "markdown",
                    "version": 1,
                    "chunk_index": 0,
                    "total_chunks": 2,
                }
            ),
        ]
        
        client.upsert(collection_name=collection_name, points=points)
        
        # Ждем, пока данные индексируются
        time.sleep(1)
        
        yield collection_name
        
    finally:
        # Удаляем тестовую коллекцию
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass


@pytest.mark.integration
def test_upload_and_search_documents(test_collection, rest_api_client):
    """Тест загрузки и поиска документов в реальной коллекции."""
    print(f"\nUsing collection: {test_collection}")
    
    # Проверяем, что коллекция содержит точки
    client = get_client()
    points_before, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
    print(f"Points in collection before upload: {len(points_before)}")
    
    # Загружаем документ
    response = rest_api_client.post(
        f"/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": "Тестовый документ для интеграционного теста",
                    "payload": {
                        "source_format": "text",
                        "category_path": "Тест / REST API"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    print(f"Upload response status: {response.status_code}")
    print(f"Upload response: {response.json()}")
    
    assert response.status_code == 200
    assert response.json()["success"] is True
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем, что точки появились
    points_after, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
    print(f"Points in collection after upload: {len(points_after)}")
    for p in points_after:
        print(f"  {p.id}: {p.payload.get('raw_text', 'N/A')[:80]}")
    
    assert len(points_after) > len(points_before), "New points should be added after upload"
    
    # Ищем документ
    response = rest_api_client.post(
        f"/v1/documents/search",
        json={
            "query_text": "интеграционный тест",
            "collection_name": test_collection,
            "limit": 5
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert "data" in response_json
    assert "results" in response_json["data"]
    assert len(response_json["data"]["results"]) > 0


@pytest.mark.integration
def test_search_categories(rest_api_client):
    """Тест поиска категорий."""
    response = rest_api_client.post(
        "/v1/categories/search",
        json={
            "query_text": "документация",
            "limit": 5
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert "results" in response_json
    assert len(response_json["results"]) > 0


@pytest.mark.integration
def test_search_categories_grouped(rest_api_client):
    """Тест группированного поиска категорий."""
    response = rest_api_client.post(
        "/v1/categories/search",
        json={
            "query_text": "документация",
            "limit": 5,
            "grouped": True
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert "results" in response_json


@pytest.mark.integration
def test_get_category_hierarchy(rest_api_client):
    """Тест получения иерархии категорий."""
    response = rest_api_client.post(
        "/v1/admin/categories/hierarchy",
        json={
            "collection_name": None,
            "depth": 1
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert "results" in response_json


@pytest.mark.integration
def test_get_collections(rest_api_client):
    """Тест получения коллекций."""
    response = rest_api_client.get("/v1/admin/collections", headers=get_auth_headers())
    
    assert response.status_code == 200
    response_json = response.json()
    assert "results" in response_json
