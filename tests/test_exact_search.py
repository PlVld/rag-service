"""Тесты гибридного exact-поиска по ключевым фразам."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.api.categories import (
    CategoryResponse,
    CategorySearchRequest,
    _default_category_item,
    search_categories,
)
from app.core.exact_search import (
    build_exact_filter,
    filter_by_min_score,
    matched_phrases,
    merge_exact_points,
    normalize_phrases,
    perform_exact_search,
    sort_hybrid_points,
)
from app.mcp_server import SearchCategoriesInput, SearchDocumentsInput, _search_documents_internal
from app.models.common import DocumentSearchRequest


def _point(point_id, score, category="A", text="текст", **payload):
    data = {
        "category_path": category,
        "raw_text": text,
        "source_id": "doc-1",
        "chunk_index": 0,
        "total_chunks": 1,
        **payload,
    }
    return qdrant_models.ScoredPoint(
        id=point_id, version=0, score=score, payload=data, vector=[0.1, 0.2, 0.3]
    )


class _FakeResponse:
    def __init__(self, points):
        self.points = points


class TestPhraseHelpers:
    def test_normalize_trims_dedups_and_limits(self):
        phrases = ["  ЕСТЬNULL ", "естьnull", "", "  ", "ISNULL / COALESCE"] + [f"p{i}" for i in range(20)]
        result = normalize_phrases(phrases)
        assert result[0] == "ЕСТЬNULL"
        assert result[1] == "ISNULL / COALESCE"
        assert len(result) == 10

    def test_none_and_empty(self):
        assert normalize_phrases(None) == []
        assert normalize_phrases([]) == []

    def test_matched_phrases_is_case_insensitive(self):
        text = "Функция ЕСТЬNULL проверяет значение"
        assert matched_phrases(text, ["естьnull"]) == ["естьnull"]

    def test_matched_phrase_respects_token_order(self):
        text = "prefix ISNULL COALESCE suffix"
        assert matched_phrases(text, ["ISNULL COALESCE"]) == ["ISNULL COALESCE"]
        assert matched_phrases(text, ["COALESCE ISNULL"]) == []

    def test_punctuation_is_a_token_boundary(self):
        text = "ISNULL / COALESCE"
        assert matched_phrases(text, ["ISNULL COALESCE"]) == ["ISNULL COALESCE"]


class TestExactFilter:
    def test_any_uses_min_should_and_base_filter(self):
        base = qdrant_models.Filter(must=[
            qdrant_models.FieldCondition(key="is_latest", match=qdrant_models.MatchValue(value=True))
        ])
        result = build_exact_filter(["ЕСТЬNULL", "ISNULL COALESCE"], "any", base)

        assert result.must == [base]
        assert result.min_should.min_count == 1
        conditions = result.min_should.conditions
        assert isinstance(conditions[0].match, qdrant_models.MatchText)
        assert isinstance(conditions[1].match, qdrant_models.MatchPhrase)

    def test_all_uses_must(self):
        result = build_exact_filter(["A", "B C"], "all")
        assert len(result.must) == 2
        assert result.min_should is None


class TestPerformExactSearch:
    @pytest.mark.asyncio
    async def test_marks_points_and_uses_query_filter(self):
        point = _point("1", 0.6, text="Вызов ЕСТЬNULL(...)")

        client = MagicMock()
        client.query_points = AsyncMock(return_value=_FakeResponse([point]))
        base = qdrant_models.Filter(must=[
            qdrant_models.FieldCondition(key="is_latest", match=qdrant_models.MatchValue(value=True))
        ])

        result = await perform_exact_search(
            client, "docs", [0.1, 0.2], base, ["естьnull"], "any", 20
        )

        kwargs = client.query_points.await_args.kwargs
        assert kwargs["query_filter"] is not None
        assert kwargs["limit"] == 20
        assert result[0].payload["_exact_match"] is True
        assert result[0].payload["_matched_phrases"] == ["естьnull"]
        assert result[0].payload["_collection_name"] == "docs"

    @pytest.mark.asyncio
    async def test_missing_text_index_returns_empty_without_raising(self):
        client = MagicMock()
        client.query_points = AsyncMock(
            side_effect=UnexpectedResponse(
                400, "Bad Request", b"index required but not found", MagicMock()
            )
        )

        result = await perform_exact_search(
            client, "docs", [0.1, 0.2], None, ["естьnull"], "any", 20
        )

        assert result == []


class TestMergeAndSort:
    def test_duplicate_is_marked_and_phrases_are_unioned(self):
        semantic = _point("1", 0.7, category="A")
        exact = _point("1", 0.0, category="A")
        exact.payload["_collection_name"] = "docs"
        exact.payload["_exact_match"] = True
        exact.payload["_matched_phrases"] = ["A"]
        semantic.payload["_collection_name"] = "docs"

        merged = merge_exact_points([semantic], [exact])

        assert len(merged) == 1
        assert merged[0].score == 0.7
        assert merged[0].payload["_exact_match"] is True
        assert merged[0].payload["_matched_phrases"] == ["A"]

    def test_missing_exact_point_is_added(self):
        semantic = _point("1", 0.7, category="A")
        semantic.payload["_collection_name"] = "docs"
        exact = _point("2", 0.4, category="B")
        exact.payload.update(_collection_name="docs", _exact_match=True, _matched_phrases=["B"])

        merged = merge_exact_points([semantic], [exact])

        assert [p.id for p in merged] == ["1", "2"]

    def test_sort_puts_exact_first_even_with_lower_score(self):
        exact = _point("1", 0.1, category="A")
        exact.payload["_exact_match"] = True
        exact.payload["_matched_phrases"] = ["X"]
        semantic = _point("2", 0.9, category="B")

        assert [p.id for p in sort_hybrid_points([semantic, exact])] == ["1", "2"]

    def test_min_score_does_not_drop_exact(self):
        exact = _point("1", 0.1)
        exact.payload["_exact_match"] = True
        semantic_low = _point("2", 0.5)
        semantic_high = _point("3", 0.9)

        result = filter_by_min_score([exact, semantic_low, semantic_high], 0.8)

        assert [p.id for p in result] == ["1", "3"]


class TestValidation:
    def test_documents_request_defaults(self):
        request = DocumentSearchRequest(query_text="q")
        assert request.exact_phrases is None
        assert request.phrase_mode == "any"

    def test_documents_request_accepts_phrases(self):
        request = DocumentSearchRequest(
            query_text="q", exact_phrases=["A", "B"], phrase_mode="all"
        )
        assert request.exact_phrases == ["A", "B"]
        assert request.phrase_mode == "all"

    def test_documents_request_rejects_more_than_10(self):
        with pytest.raises(Exception):
            DocumentSearchRequest(query_text="q", exact_phrases=[f"p{i}" for i in range(11)])

    def test_mcp_documents_input(self):
        data = SearchDocumentsInput(query_text="q", exact_phrases=["A"], phrase_mode="all")
        assert data.exact_phrases == ["A"]
        assert data.phrase_mode == "all"

    def test_mcp_categories_input(self):
        data = SearchCategoriesInput(query_text="q", exact_phrases=["A"])
        assert data.exact_phrases == ["A"]
        assert data.phrase_mode == "any"

    def test_categories_request(self):
        data = CategorySearchRequest(query_text="q", exact_phrases=["A"], phrase_mode="all")
        assert data.exact_phrases == ["A"]
        assert data.phrase_mode == "all"


class TestCategoriesExact:
    def test_default_output_marks_exact_only_when_requested(self):
        cat = CategoryResponse(
            id="1", score=0.5, category_name="API", category_path="Документация / API",
            categories=["Документация", "API"], category_level=1,
            exact_match=True, matched_phrases=["API"],
        )

        assert _default_category_item(cat, True) == {
            "path": "Документация / API",
            "score": 0.5,
            "exact_match": True,
            "matched_phrases": ["API"],
        }
        assert _default_category_item(cat, False) == {
            "path": "Документация / API",
            "score": 0.5,
        }

    @pytest.mark.asyncio
    async def test_exact_first_and_threshold_bypass(self):
        plain = qdrant_models.ScoredPoint(
            id="1", version=0, score=0.9,
            payload={"category_path": "Синтаксис / Таблицы", "category_name": "Таблицы",
                     "category_level": 1, "category_id": "1"},
        )
        exact = qdrant_models.ScoredPoint(
            id="2", version=0, score=0.5,
            payload={"category_path": "API / Методы", "category_name": "Методы",
                     "category_level": 1, "category_id": "2"},
        )
        client = MagicMock()
        client.query_points = AsyncMock(return_value=_FakeResponse([plain, exact]))

        with patch("app.api.categories.get_client", return_value=client), \
             patch("app.api.categories.encode_text", new_callable=AsyncMock, return_value=[0.1] * 3):
            results = await search_categories(
                "методы API", 5, min_score=0.99,
                exact_phrases=["API"], phrase_mode="any",
            )

        assert [cat.category_path for cat in results] == ["API / Методы"]
        assert results[0].exact_match is True
        assert results[0].matched_phrases == ["API"]
        assert client.query_points.await_args.kwargs["limit"] == 25
        assert client.query_points.await_args.kwargs["score_threshold"] is None

    @pytest.mark.asyncio
    async def test_all_mode_requires_every_phrase(self):
        partial = qdrant_models.ScoredPoint(
            id="1", version=0, score=0.9,
            payload={"category_path": "API / Методы", "category_name": "Методы",
                     "category_level": 1, "category_id": "1"},
        )
        full = qdrant_models.ScoredPoint(
            id="2", version=0, score=0.4,
            payload={"category_path": "Документация / API", "category_name": "API",
                     "category_level": 1, "category_id": "2"},
        )
        client = MagicMock()
        client.query_points = AsyncMock(return_value=_FakeResponse([partial, full]))

        with patch("app.api.categories.get_client", return_value=client), \
             patch("app.api.categories.encode_text", new_callable=AsyncMock, return_value=[0.1] * 3):
            results = await search_categories(
                "документация API", 5, exact_phrases=["Документация", "API"],
                phrase_mode="all",
            )

        assert results[0].category_path == "Документация / API"
        assert results[0].exact_match is True
        assert results[1].exact_match is None


class TestInternalExactFlow:
    @pytest.mark.asyncio
    async def test_exact_results_first_and_survive_min_score(self):
        prepared = {
            "query_vector": [0.1, 0.2, 0.3],
            "base_filter": None,
            "cat_ids": [],
            "category_path_filter": None,
        }
        semantic = _point("1", 0.9, category="A")
        exact = _point("2", 0.2, category="B")
        exact.payload.update(
            _collection_name="test", _exact_match=True, _matched_phrases=["ЕСТЬNULL"]
        )

        with patch(
            "app.mcp_server._prepare_search_metadata",
            new_callable=AsyncMock,
            return_value={"success": False, "data": {}, "prepared": prepared},
        ), patch("app.mcp_server.get_client", return_value=MagicMock()), \
           patch(
               "app.mcp_server._perform_search",
               new_callable=AsyncMock,
               return_value=[semantic],
           ), patch(
               "app.mcp_server.perform_exact_search",
               new_callable=AsyncMock,
               return_value=[exact],
           ) as mock_exact, patch(
               "app.mcp_server.rerank_by_query_similarity",
               new_callable=AsyncMock,
               side_effect=lambda points, vector: points,
           ):
            result = await _search_documents_internal(
                query_text="ЕСТЬNULL",
                collection_name="test",
                limit=5,
                group=False,
                min_score=0.95,
                exact_phrases=["ЕСТЬNULL"],
            )

        results = result["data"]["results"]
        assert [r["category_path"] for r in results] == ["B"]
        assert results[0]["exact_match"] is True
        assert results[0]["matched_phrases"] == ["ЕСТЬNULL"]
        assert mock_exact.await_args.args[3] is None  # base_filter
        assert mock_exact.await_args.args[4] == ["ЕСТЬNULL"]
