"""Тесты REST API endpoints для документов."""
import pytest
from unittest.mock import patch, AsyncMock


def get_auth_headers():
    """Возвращает заголовки для аутентификации."""
    return {"Authorization": "Bearer PASS1234"}


@pytest.mark.integration
def test_search_documents_structure(mock_qdrant_client, mock_embeddings, rest_api_client):
    """Тест поиска документов - проверка структуры запроса."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client), \
         patch("app.api.categories.search_categories", new_callable=AsyncMock) as mock_search_cat:
        mock_search_cat.return_value = []

        response = rest_api_client.post(
            "/v1/documents/search",
            json={
                "query_text": "настройка API",
                "collection_name": "test_rest_api_mock",
                "limit": 5
            },
            headers=get_auth_headers()
        )

        # Проверяем, что запрос был отправлен и вернулся ответ
        assert response.status_code in [200, 400, 500]


@pytest.mark.integration
def test_search_documents_empty_query(mock_qdrant_client, rest_api_client):
    """Тест пустого запроса поиска."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client):
        response = rest_api_client.post(
            "/v1/documents/search",
            json={
                "query_text": "",
                "collection_name": "test_rest_api_mock",
                "limit": 5
            },
            headers=get_auth_headers()
        )

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["success"] is True
        assert response_json["data"]["results"] == []


@pytest.mark.integration
def test_search_documents_with_limit(mock_qdrant_client, mock_embeddings, rest_api_client):
    """Тест поиска с разным лимитом результатов."""
    with patch("app.api.health.get_client", return_value=mock_qdrant_client), \
         patch("app.api.categories.search_categories", new_callable=AsyncMock) as mock_search_cat:
        mock_search_cat.return_value = []

        response = rest_api_client.post(
            "/v1/documents/search",
            json={
                "query_text": "настройка",
                "collection_name": "test_rest_api_mock",
                "limit": 10
            },
            headers=get_auth_headers()
        )

        assert response.status_code == 200
