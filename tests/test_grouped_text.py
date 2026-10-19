"""Тесты сгруппированного текста чанков (REST и MCP)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from qdrant_client.http import models as qdrant_models

from app.api.documents_grouped import get_grouped_text
from app.mcp.tools import _format_chunk_numbers, get_grouped_text_tool
from app.mcp_server import GetGroupedTextInput


def make_point(payload):
    return SimpleNamespace(payload=payload)


class FakeQdrantClient:
    """Минимальный клиент Qdrant: отдаёт точки страницами и пишет вызовы."""

    def __init__(self, points):
        self.points = list(points)
        self.scroll_calls = []

    async def scroll(
        self,
        collection_name,
        scroll_filter,
        limit,
        offset,
        with_payload,
        with_vectors,
    ):
        self.scroll_calls.append({
            "collection_name": collection_name,
            "scroll_filter": scroll_filter,
            "limit": limit,
            "offset": offset,
            "with_payload": with_payload,
            "with_vectors": with_vectors,
        })
        start = offset or 0
        page = self.points[start:start + limit]
        next_offset = start + len(page) if start + len(page) < len(self.points) else None
        return page, next_offset


@pytest.fixture
def grouped_points():
    return [
        make_point({
            "raw_text": "c0",
            "source_id": "doc-1",
            "version": 1,
            "category_path": "A / B",
            "chunk_index": 0,
            "total_chunks": 5,
            "original_filename": "doc1.md",
        }),
        make_point({
            "raw_text": "c1",
            "source_id": "doc-1",
            "version": 1,
            "category_path": "A / B",
            "chunk_index": 1,
            "total_chunks": 5,
            "original_filename": "doc1.md",
        }),
        make_point({
            "raw_text": "c2",
            "source_id": "doc-1",
            "version": 1,
            "category_path": "A / B / C",
            "chunk_index": 2,
            "total_chunks": 5,
            "original_filename": "doc1.md",
        }),
        make_point({
            "raw_text": "d0",
            "source_id": "doc-2",
            "version": 1,
            "category_path": "A / B",
            "chunk_index": 0,
            "total_chunks": 2,
            "original_filename": "doc2.md",
        }),
    ]


class TestGetGroupedText:
    @pytest.mark.asyncio
    async def test_groups_by_document_and_category(self, grouped_points):
        client = FakeQdrantClient(grouped_points)
        with patch("app.api.documents_grouped.get_client", return_value=client):
            results = await get_grouped_text(
                collection_name="documents",
                categories=["A / B"],
            )

        assert len(results) == 3
        # doc-1 / A / B: два чанка склеены по порядку
        first = results[0]
        assert first["source_id"] == "doc-1"
        assert first["category_path"] == "A / B"
        assert first["document"] == "c0\nc1"
        assert first["chunk_indices"] == [0, 1]
        assert first["chunk_count"] == 2
        assert first["total_chunks"] == 5

        # Вложенная категория A / B / C тоже попала в выборку
        assert results[1]["category_path"] == "A / B / C"
        assert results[1]["chunk_indices"] == [2]

        assert results[2]["source_id"] == "doc-2"

    @pytest.mark.asyncio
    async def test_documents_filter_is_passed_to_qdrant(self, grouped_points):
        client = FakeQdrantClient(grouped_points)
        with patch("app.api.documents_grouped.get_client", return_value=client):
            await get_grouped_text(
                collection_name="documents",
                documents=["doc-1"],
            )

        scroll_filter = client.scroll_calls[0]["scroll_filter"]
        conditions = {condition.key: condition.match for condition in scroll_filter.must}

        assert conditions["is_latest"].value is True
        assert conditions["source_id"].any == ["doc-1"]

    @pytest.mark.asyncio
    async def test_categories_filter_is_pushed_to_qdrant(self, grouped_points):
        client = FakeQdrantClient(grouped_points)
        with patch("app.api.documents_grouped.get_client", return_value=client):
            await get_grouped_text(
                collection_name="documents",
                categories=["A/B"],
            )

        scroll_filter = client.scroll_calls[0]["scroll_filter"]
        category_filters = [
            condition
            for condition in scroll_filter.should
            if isinstance(condition, qdrant_models.Filter)
        ]
        level_conditions = {
            condition.key: condition.match.value
            for condition in category_filters[0].must
            if condition.key.startswith("category_level")
        }
        assert level_conditions == {"category_level0": "A", "category_level1": "B"}

    @pytest.mark.asyncio
    async def test_metadata_scan_does_not_fetch_raw_text(self, grouped_points):
        client = FakeQdrantClient(grouped_points)
        with patch("app.api.documents_grouped.get_client", return_value=client):
            await get_grouped_text(collection_name="documents")

        # Первая фаза — только метаданные, без raw_text.
        metadata_calls = [
            call for call in client.scroll_calls
            if "raw_text" not in call["with_payload"]
        ]
        assert metadata_calls
        assert "source_id" in metadata_calls[0]["with_payload"]
        assert metadata_calls[0]["with_vectors"] is False
        assert metadata_calls[0]["limit"] > 1000

        # Вторая фаза — полный текст только по выбранным группам.
        text_calls = [
            call for call in client.scroll_calls
            if "raw_text" in call["with_payload"]
        ]
        assert text_calls
        assert text_calls[0]["scroll_filter"].must is not None

    @pytest.mark.asyncio
    async def test_scroll_uses_pagination(self, grouped_points, monkeypatch):
        monkeypatch.setattr("app.api.documents_grouped._SCROLL_PAGE_SIZE", 2)
        client = FakeQdrantClient(grouped_points)
        with patch("app.api.documents_grouped.get_client", return_value=client):
            results = await get_grouped_text(collection_name="documents")

        metadata_calls = [
            call for call in client.scroll_calls
            if "raw_text" not in call["with_payload"]
        ]
        assert len(metadata_calls) == 2
        assert metadata_calls[1]["offset"] == 2
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_include_old_versions_removes_latest_filter(self, grouped_points):
        client = FakeQdrantClient(grouped_points)
        with patch("app.api.documents_grouped.get_client", return_value=client):
            await get_grouped_text(
                collection_name="documents",
                include_old_versions=True,
            )

        assert client.scroll_calls[0]["scroll_filter"] is None

    @pytest.mark.asyncio
    async def test_max_text_length_and_limit(self, grouped_points):
        client = FakeQdrantClient(grouped_points)
        with patch("app.api.documents_grouped.get_client", return_value=client):
            results = await get_grouped_text(
                collection_name="documents",
                max_text_length=2,
                limit=1,
            )

        assert len(results) == 1
        assert results[0]["document"] == "c0"


class TestGroupedTextRestEndpoint:
    def test_returns_grouped_results(self, rest_api_client, grouped_points):
        client = FakeQdrantClient(grouped_points)
        with patch("app.api.documents_grouped.get_client", return_value=client):
            response = rest_api_client.post(
                "/v1/documents/grouped-text",
                json={
                    "collection_name": "documents",
                    "categories": ["A / B"],
                },
                headers={"Authorization": "Bearer PASS1234"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        results = body["data"]["results"]
        assert len(results) == 3
        assert results[0]["source_id"] == "doc-1"
        assert results[0]["document"] == "c0\nc1"
        assert results[0]["chunk_indices"] == [0, 1]


class TestGroupedTextMcpTool:
    @pytest.mark.asyncio
    async def test_output_contains_metadata_and_text(self):
        internal_result = {
            "success": True,
            "data": {
                "results": [{
                    "source_id": "doc-1",
                    "collection_name": "documents",
                    "category_path": "A / B",
                    "version": 1,
                    "first_chunk_index": 0,
                    "last_chunk_index": 2,
                    "chunk_indices": [0, 1, 2],
                    "chunk_count": 3,
                    "total_chunks": 5,
                    "original_filename": "doc1.md",
                    "document": "c0\nc1\nc2",
                }]
            },
        }

        with patch(
            "app.mcp.tools._get_grouped_text_internal",
            new_callable=AsyncMock,
            return_value=internal_result,
        ):
            output = await get_grouped_text_tool(
                GetGroupedTextInput(collection_name="documents")
            )

        assert "Найдено 1 групп" in output
        assert "source_id=doc-1" in output
        assert "collection_name=documents" in output
        assert "category_path=A / B" in output
        assert "чанки 0–2 из 5" in output
        assert "c0\nc1\nc2" in output

    def test_chunk_numbers_compresses_gaps(self):
        assert _format_chunk_numbers([0, 1, 2, 5, 7, 8], 10) == "чанки 0–2, 5, 7–8 из 10"
        assert _format_chunk_numbers([3], 7) == "чанк 3 из 7"
        assert _format_chunk_numbers([], 4) == "всего чанков: 4"
