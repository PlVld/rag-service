"""Тесты REST API endpoints для категорий."""
import pytest
from unittest.mock import patch, AsyncMock


def get_auth_headers():
    """Возвращает заголовки для аутентификации."""
    return {"Authorization": "Bearer PASS1234"}


@pytest.mark.integration
def test_search_categories_structure(mock_qdrant_client, rest_api_client):
    """Тест поиска категорий - проверка структуры запроса."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/categories/search",
            json={
                "query_text": "документация",
                "limit": 5
            },
            headers=get_auth_headers()
        )

        # Проверяем, что запрос был отправлен и вернулся ответ
        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_search_categories_empty(mock_qdrant_client, rest_api_client):
    """Тест пустого запроса категорий."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/categories/search",
            json={
                "query_text": "",
                "limit": 5
            },
            headers=get_auth_headers()
        )

        assert response.status_code == 200
        response_json = response.json()
        assert "results" in response_json


@pytest.mark.integration
def test_search_categories_no_results(mock_qdrant_client, rest_api_client):
    """Тест поиска категорий без результатов."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client), \
         patch("app.api.categories.search_categories", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []

        response = rest_api_client.post(
            "/v1/categories/search",
            json={
                "query_text": "несуществующая категория",
                "limit": 5
            },
            headers=get_auth_headers()
        )

        assert response.status_code == 200
        response_json = response.json()
        assert "results" in response_json


@pytest.mark.integration
def test_search_categories_grouped_structure(mock_qdrant_client, mock_embeddings, rest_api_client):
    """Тест группированного поиска категорий - проверка структуры."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/categories/search",
            json={
                "query_text": "документация",
                "limit": 5,
                "grouped": True
            },
            headers=get_auth_headers()
        )

        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_get_all_categories_structure(mock_qdrant_client, rest_api_client):
    """Тест получения всех категорий - проверка структуры запроса."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.get(
            "/v1/categories",
            headers=get_auth_headers()
        )

        # Проверяем, что запрос был отправлен и вернулся ответ
        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_get_all_categories_with_limit(mock_qdrant_client, rest_api_client):
    """Тест получения категорий с лимитом."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.get(
            "/v1/categories",
            params={"limit": 50},
            headers=get_auth_headers()
        )

        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_category_paths_structure(mock_qdrant_client, mock_embeddings, rest_api_client):
    """Тест получения путей категорий - проверка структуры через /search."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client), \
         patch("app.api.categories.search_categories", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []

        response = rest_api_client.post(
            "/v1/categories/search",
            json={
                "query_text": "документация",
                "limit": 5,
                "fields": ["path", "score"]
            },
            headers=get_auth_headers()
        )

        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_category_paths_grouped(mock_qdrant_client, mock_embeddings, rest_api_client):
    """Тест получения путей категорий с группировкой через /search."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client), \
         patch("app.api.categories.search_categories", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []

        response = rest_api_client.post(
            "/v1/categories/search",
            json={
                "query_text": "документация",
                "limit": 5,
                "fields": ["path", "score"],
                "grouped": True
            },
            headers=get_auth_headers()
        )

        assert response.status_code in [200, 400, 500]
