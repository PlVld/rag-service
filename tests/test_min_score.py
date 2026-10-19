"""Тесты необязательного порога релевантности min_score."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from qdrant_client import models as qdrant_models

from app.api.categories import CategorySearchRequest, search_categories
from app.api.documents_search import _perform_search
from app.mcp_server import (
    SearchCategoriesInput,
    SearchDocumentsInput,
    _search_documents_internal,
)
from app.models.common import DocumentSearchRequest


class _FakeResponse:
    def __init__(self, points):
        self.points = points


def _point(score: float, category: str) -> qdrant_models.ScoredPoint:
    return qdrant_models.ScoredPoint(
        id=category,
        version=0,
        score=score,
        payload={
            "category_path": category,
            "raw_text": "текст",
            "source_id": "doc-1",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    )


class TestMinScoreValidation:
    """Валидация параметра min_score во входных моделях."""

    def test_documents_request_default_is_055(self):
        request = DocumentSearchRequest(query_text="запрос")
        assert request.min_score == 0.55

    def test_documents_request_null_uses_default(self):
        assert DocumentSearchRequest(query_text="q", min_score=None).min_score == 0.55

    def test_mcp_null_uses_default(self):
        assert SearchDocumentsInput(query_text="q", min_score=None).min_score == 0.55
        assert SearchCategoriesInput(query_text="q", min_score=None).min_score == 0.55

    def test_categories_request_null_uses_default(self):
        assert CategorySearchRequest(query_text="q", min_score=None).min_score == 0.55

    def test_documents_request_accepts_bounds(self):
        assert DocumentSearchRequest(query_text="q", min_score=0).min_score == 0
        assert DocumentSearchRequest(query_text="q", min_score=1).min_score == 1

    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_documents_request_rejects_out_of_range(self, value):
        with pytest.raises(Exception):
            DocumentSearchRequest(query_text="q", min_score=value)

    def test_mcp_documents_input_default_and_value(self):
        assert SearchDocumentsInput(query_text="q").min_score == 0.55
        assert SearchDocumentsInput(query_text="q", min_score=0.5).min_score == 0.5

    def test_mcp_categories_input_default_and_value(self):
        assert SearchCategoriesInput(query_text="q").min_score == 0.55
        assert SearchCategoriesInput(query_text="q", min_score=0.5).min_score == 0.5

    def test_categories_request_default_and_value(self):
        assert CategorySearchRequest(query_text="q").min_score == 0.55
        assert CategorySearchRequest(query_text="q", min_score=0.5).min_score == 0.5


class TestPerformSearchThreshold:
    """score_threshold доходит до Qdrant в обоих режимах поиска."""

    @pytest.mark.asyncio
    async def test_passed_without_categories(self):
        client = MagicMock()
        client.query_points = AsyncMock(return_value=_FakeResponse([]))

        await _perform_search(
            client, "test", [0.1] * 3, None, [], 5, None, 0.5
        )

        assert client.query_points.await_args.kwargs["score_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_passed_to_both_prefetches_in_rrf_mode(self):
        client = MagicMock()
        client.query_points = AsyncMock(return_value=_FakeResponse([]))

        await _perform_search(
            client, "test", [0.1] * 3, None, ["cat-id"], 5, None, 0.4
        )

        prefetch = client.query_points.await_args.kwargs["prefetch"]
        assert [p.score_threshold for p in prefetch] == [0.4, 0.4]

    @pytest.mark.asyncio
    async def test_none_means_no_threshold(self):
        client = MagicMock()
        client.query_points = AsyncMock(return_value=_FakeResponse([]))

        await _perform_search(client, "test", [0.1] * 3, None, [], 5)

        assert client.query_points.await_args.kwargs["score_threshold"] is None


class TestCategoriesThreshold:
    """Порог применяется при поиске категорий."""

    @pytest.mark.asyncio
    async def test_score_threshold_passed_to_qdrant(self):
        point = MagicMock()
        point.id = "cat-id"
        point.score = 0.9
        point.payload = {
            "category_path": "A / B",
            "category_name": "B",
            "category_level": 1,
        }
        client = MagicMock()
        client.query_points = AsyncMock(return_value=_FakeResponse([point]))

        with patch("app.api.categories.get_client", return_value=client), \
             patch(
                 "app.api.categories.encode_text",
                 new_callable=AsyncMock,
                 return_value=[0.1] * 3,
             ):
            results = await search_categories("запрос", 5, min_score=0.5)

        assert client.query_points.await_args.kwargs["score_threshold"] == 0.5
        assert len(results) == 1


class TestInternalFiltering:
    """После reranking результаты ниже порога отбрасываются."""

    @pytest.mark.asyncio
    async def test_results_below_min_score_are_dropped(self):
        prepared = {
            "query_vector": [0.1] * 3,
            "base_filter": None,
            "cat_ids": [],
            "category_path_filter": None,
        }
        points = [_point(0.9, "A"), _point(0.5, "B"), _point(0.2, "C")]

        with patch(
            "app.mcp_server._prepare_search_metadata",
            new_callable=AsyncMock,
            return_value={"success": False, "data": {}, "prepared": prepared},
        ), patch("app.mcp_server.get_client", return_value=MagicMock()), \
           patch(
               "app.mcp_server._perform_search",
               new_callable=AsyncMock,
               return_value=points,
           ) as mock_search, patch(
               "app.mcp_server.rerank_by_query_similarity",
               new_callable=AsyncMock,
               return_value=points,
           ):
            result = await _search_documents_internal(
                query_text="запрос",
                collection_name="test",
                limit=10,
                group=False,
                min_score=0.5,
            )

        scores = [r["score"] for r in result["data"]["results"]]
        assert scores == [0.9, 0.5]
        # Порог передан в поиск
        assert mock_search.await_args.args[-1] == 0.5
