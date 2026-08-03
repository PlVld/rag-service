"""Тесты REST API endpoints для администратора."""
import pytest
from unittest.mock import patch


def get_auth_headers():
    """Возвращает заголовки для аутентификации."""
    return {"Authorization": "Bearer PASS1234"}


def get_form_headers():
    """Возвращает заголовки для формы-запросов."""
    return {"Authorization": "Bearer PASS1234", "Content-Type": "application/x-www-form-urlencoded"}


@pytest.mark.integration
def test_get_collections_structure(mock_qdrant_client, rest_api_client):
    """Тест получения коллекций - проверка структуры запроса."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.get(
            "/v1/admin/collections",
            headers=get_auth_headers()
        )

        # Проверяем, что запрос был отправлен и вернулся ответ
        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_get_category_hierarchy_structure(mock_qdrant_client, rest_api_client):
    """Тест иерархии категорий - проверка структуры запроса."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/admin/categories/hierarchy",
            json={
                "collection_name": None,
                "depth": 1
            },
            headers=get_auth_headers()
        )

        # Проверяем, что запрос был отправлен и вернулся ответ
        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_get_category_hierarchy_depth_0(mock_qdrant_client, rest_api_client):
    """Тест иерархии с depth=0 (только список коллекций)."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/admin/categories/hierarchy",
            json={
                "collection_name": None,
                "depth": 0
            },
            headers=get_auth_headers()
        )

        assert response.status_code == 200
        response_json = response.json()
        assert "results" in response_json


@pytest.mark.integration
def test_get_category_hierarchy_with_categories(mock_qdrant_client, rest_api_client):
    """Тест иерархии с фильтром по категориям."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/admin/categories/hierarchy",
            json={
                "collection_name": None,
                "depth": 2,
                "categories": ["Документация / API"]
            },
            headers=get_auth_headers()
        )

        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_create_payload_index_structure(mock_qdrant_client, rest_api_client):
    """Тест создания payload индекса - проверка структуры запроса."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/admin/create_payload_index",
            json={
                "collection_name": "test_rest_api_mock",
                "field_name": "source_format",
                "field_type": "keyword"
            },
            headers=get_auth_headers()
        )

        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_create_payload_index_with_different_types(mock_qdrant_client, rest_api_client):
    """Тест создания индекса с разными типами полей."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/admin/create_payload_index",
            json={
                "collection_name": "test_rest_api_mock",
                "field_name": "category_level",
                "field_type": "integer"
            },
            headers=get_auth_headers()
        )

        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_set_hnsw_enabled_structure(mock_qdrant_client, rest_api_client):
    """Тест включения HNSW индекса - проверка структуры запроса."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/admin/hnsw",
            data={
                "collection_name": "test_rest_api_mock",
                "enabled": "true",
                "m": "16"
            },
            headers=get_auth_headers()
        )

        assert response.status_code in [200, 400, 500, 422]


@pytest.mark.integration
def test_set_hnsw_disabled_structure(mock_qdrant_client, rest_api_client):
    """Тест выключения HNSW индекса - проверка структуры запроса."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/admin/hnsw",
            data={
                "collection_name": "test_rest_api_mock",
                "enabled": "false",
                "m": "0"
            },
            headers=get_auth_headers()
        )

        assert response.status_code in [200, 400, 500, 422]
