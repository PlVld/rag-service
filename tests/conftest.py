import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import time
import uuid


@pytest.fixture
def mock_embeddings():
    """Мок для функции encode_text."""
    with patch("app.core.embeddings.encode_text", new_callable=AsyncMock) as mock:
        mock.return_value = [[0.1] * 384]  # Список списков для пакетной обработки
        yield mock


@pytest.fixture
def mock_search_categories():
    """Мок для функции search_categories."""
    with patch("app.api.categories.search_categories", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def mock_search_categories_by_collections():
    """Мок для функции search_categories_by_collections."""
    with patch("app.api.categories.search_categories_by_collections", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def mock_search_points():
    """Мок для результатов поиска в Qdrant."""
    points = [
        MagicMock(
            id="point-1",
            score=0.95,
            payload={
                "raw_text": "Текст документа по настройке API",
                "source_id": "doc-1",
                "source_format": "text",
                "is_latest": True,
                "category_path": "Документация / API",
                "original_filename": "api_guide.md",
                "content_type": "markdown",
                "version": 1,
                "file_path": "/docs/api/api_guide.md",
                "created_at": "2025-01-01T00:00:00Z",
                "chunk_index": 0,
                "total_chunks": 5,
            }
        ),
        MagicMock(
            id="point-2",
            score=0.85,
            payload={
                "raw_text": "Настройка параметров API",
                "source_id": "doc-2",
                "source_format": "text",
                "is_latest": True,
                "category_path": "Руководства / Настройка",
                "original_filename": "setup_guide.md",
                "content_type": "markdown",
                "version": 1,
                "file_path": "/docs/setup/setup_guide.md",
                "created_at": "2025-01-02T00:00:00Z",
                "chunk_index": 1,
                "total_chunks": 8,
            }
        ),
        MagicMock(
            id="point-3",
            score=0.75,
            payload={
                "raw_text": "API конфигурация",
                "source_id": "doc-3",
                "source_format": "text",
                "is_latest": True,
                "category_path": "Справочник / Конфигурация",
                "original_filename": "config_ref.md",
                "content_type": "markdown",
                "version": 1,
                "file_path": "/docs/config/config_ref.md",
                "created_at": "2025-01-03T00:00:00Z",
                "chunk_index": 2,
                "total_chunks": 3,
            }
        ),
    ]
    yield points


@pytest.fixture
def mock_get_all_chunks():
    """Мок для функции get_all_chunks."""
    with patch("app.api.documents.get_all_chunks", new_callable=AsyncMock) as mock:
        chunks = [
            MagicMock(payload={"raw_text": f"Чанк {i}"}) for i in range(10)
        ]
        mock.return_value = chunks
        yield mock


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Мок для настроек приложения, чтобы использовать тестовую коллекцию."""
    monkeypatch.setattr("app.core.config.settings.default_collection", "test")


@pytest.fixture(autouse=True)
def reset_qdrant_client():
    """Сброс глобального кэша клиента Qdrant перед каждым тестом."""
    from app.api import health
    health._qdrant_client = None
    yield


@pytest.fixture
def mock_collections():
    """Мок для получения списка коллекций."""
    with patch("qdrant_client.QdrantClient.get_collections") as mock:
        mock.return_value = type('obj', (object,), {
            'collections': [
                type('obj', (object,), {'name': 'documents'})(),
                type('obj', (object,), {'name': 'categories'})(),
            ]
        })()
        yield mock


@pytest.fixture
def mock_facet_result():
    """Мок для facet-запроса."""
    with patch("qdrant_client.QdrantClient.facet") as mock:
        mock.return_value = type('obj', (object,), {
            'hits': [
                type('obj', (object,), {'value': 'Документация / API', 'count': 10})(),
                type('obj', (object,), {'value': 'Документация / Настройка', 'count': 5})(),
            ]
        })()
        yield mock


@pytest.fixture
def mock_qdrant_client():
    """Мок для клиента Qdrant с необходимыми методами для REST API тестов."""
    mock_client = MagicMock()
    
    # Mock get_collections
    mock_client.get_collections.return_value = type('obj', (object,), {
        'collections': [
            type('obj', (object,), {'name': 'documents'})(),
            type('obj', (object,), {'name': 'categories'})(),
            type('obj', (object,), {'name': 'test_rest_api_mock'})(),
        ]
    })()
    
    # Mock query_points (для поиска)
    def mock_query_points(collection_name, query, limit, with_payload=True, filter=None):
        points = [
            MagicMock(
                id="doc-1-point-1",
                score=0.95,
                payload={
                    "raw_text": "Текст документа по настройке API",
                    "source_id": "doc-1",
                    "source_format": "text",
                    "is_latest": True,
                    "category_path": "Документация / API",
                    "original_filename": "api_guide.md",
                    "content_type": "markdown",
                    "version": 1,
                    "file_path": "/docs/api/api_guide.md",
                    "created_at": "2025-01-01T00:00:00Z",
                    "chunk_index": 0,
                    "total_chunks": 5,
                }
            ),
            MagicMock(
                id="doc-2-point-1",
                score=0.85,
                payload={
                    "raw_text": "Настройка параметров API",
                    "source_id": "doc-2",
                    "source_format": "text",
                    "is_latest": True,
                    "category_path": "Руководства / Настройка",
                    "original_filename": "setup_guide.md",
                    "content_type": "markdown",
                    "version": 1,
                    "file_path": "/docs/setup/setup_guide.md",
                    "created_at": "2025-01-02T00:00:00Z",
                    "chunk_index": 0,
                    "total_chunks": 8,
                }
            ),
            MagicMock(
                id="doc-3-point-1",
                score=0.75,
                payload={
                    "raw_text": "API конфигурация",
                    "source_id": "doc-3",
                    "source_format": "text",
                    "is_latest": True,
                    "category_path": "Справочник / Конфигурация",
                    "original_filename": "config_ref.md",
                    "content_type": "markdown",
                    "version": 1,
                    "file_path": "/docs/config/config_ref.md",
                    "created_at": "2025-01-03T00:00:00Z",
                    "chunk_index": 0,
                    "total_chunks": 3,
                }
            ),
        ]
        # Применяем фильтр, если он задан
        if filter:
            # Упрощенная обработка фильтра для тестов
            if hasattr(filter, 'must'):
                for condition in filter.must:
                    if hasattr(condition, 'key') and condition.key == "is_latest":
                        if hasattr(condition.match, 'value') and condition.match.value is True:
                            points = [p for p in points if p.payload.get("is_latest") is True]
        
        return type('obj', (object,), {
            'points': points[:limit]
        })()
    
    mock_client.query_points = mock_query_points
    
    # Mock scroll (для получения всех точек)
    def mock_scroll(collection_name, limit=100, with_payload=True, scroll_filter=None):
        points = [
            MagicMock(
                id="scroll-doc-1",
                score=None,
                payload={
                    "raw_text": "Полный текст документа",
                    "source_id": "scroll-doc-1",
                    "source_format": "text",
                    "is_latest": True,
                    "category_path": "Документация / API",
                    "original_filename": "full_doc.md",
                    "content_type": "markdown",
                    "version": 1,
                    "chunk_index": 0,
                    "total_chunks": 3,
                }
            ),
            MagicMock(
                id="scroll-doc-2",
                score=None,
                payload={
                    "raw_text": "Текст второго документа",
                    "source_id": "scroll-doc-2",
                    "source_format": "text",
                    "is_latest": True,
                    "category_path": "Документация / Настройка",
                    "original_filename": "second_doc.md",
                    "content_type": "markdown",
                    "version": 1,
                    "chunk_index": 0,
                    "total_chunks": 2,
                }
            ),
        ]
        return (points, None)
    
    mock_client.scroll = mock_scroll
    
    # Mock facet
    def mock_facet(collection_name, key, limit=10, facet_filter=None):
        return type('obj', (object,), {
            'hits': [
                type('obj', (object,), {'value': 'Документация / API', 'count': 10})(),
                type('obj', (object,), {'value': 'Документация / Настройка', 'count': 5})(),
            ]
        })()
    
    mock_client.facet = mock_facet
    
    # Mock retrieve
    def mock_retrieve(collection_name, ids, with_payload=True):
        points = [
            MagicMock(
                id=doc_id,
                payload={
                    "raw_text": f"Текст для {doc_id}",
                    "source_id": doc_id,
                    "source_format": "text",
                    "is_latest": True,
                    "category_path": "Документация / API",
                    "original_filename": "retrieved_doc.md",
                    "content_type": "markdown",
                    "version": 1,
                }
            ) for doc_id in ids
        ]
        return points
    
    mock_client.retrieve = mock_retrieve
    
    # Mock upsert
    mock_client.upsert.return_value = None
    
    # Mock create_collection
    mock_client.create_collection.return_value = None
    
    # Mock delete_collection
    mock_client.delete_collection.return_value = None
    
    return mock_client


@pytest.fixture
def mock_categories():
    """Мок для результатов поиска категорий."""
    with patch("app.api.categories.search_categories", new_callable=AsyncMock) as mock:
        mock.return_value = [
            MagicMock(
                id="cat-1",
                score=0.95,
                category_name="API",
                category_path="Документация / API",
                categories=["Документация", "API"],
                category_level=1,
                category_id="cat-id-1",
                levels={"category_level0": "Документация", "category_level1": "API"},
                id_levels={"category_id_level0": "id-0", "category_id_level1": "id-1"},
            ),
        ]
        yield mock


@pytest.fixture
def test_collection():
    """Создает и удаляет тестовую коллекцию для интеграционных тестов."""
    from app.api.health import get_client
    
    collection_name = f"test_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
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
        
        # Заполняем коллекцию тестовыми данными
        from qdrant_client.http import models as qdrant_models
        
        # Генерируем тестовые эмбеддинги (1024-мерные векторы для BAAI/bge-m3)
        import random
        random.seed(42)
        
        points = [
            qdrant_models.PointStruct(
                id=uuid.uuid4().hex,
                vector=[random.gauss(0, 1) for _ in range(1024)],
                payload={
                    "raw_text": "Тестовый документ по настройке API",
                    "source_id": "test-doc-1",
                    "source_format": "text",
                    "is_latest": True,
                    "category_path": "Документация / API",
                    "original_filename": "test_api.md",
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
                    "raw_text": "Пример использования API для разработчиков",
                    "source_id": "test-doc-2",
                    "source_format": "text",
                    "is_latest": True,
                    "category_path": "Документация / Примеры",
                    "original_filename": "test_examples.md",
                    "content_type": "markdown",
                    "version": 1,
                    "chunk_index": 0,
                    "total_chunks": 2,
                }
            ),
        ]
        
        client.upsert(collection_name=collection_name, points=points)
        
        # Ждем индексации
        time.sleep(1)
        
        yield collection_name
        
    finally:
        # Удаляем тестовую коллекцию
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass


@pytest.fixture
def rest_api_client():
    """TestClient для REST API тестов."""
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app, raise_server_exceptions=True)
    return client
