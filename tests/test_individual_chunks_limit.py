"""Регрессия: при group=false итоговый limit соблюдается после дедупликации.

Кандидатов для поиска запрашивается с запасом (limit * 2), поэтому без
финального среза возвращалось до 2 * limit результатов.
"""
import pytest
from qdrant_client import models as qdrant_models

from app.api.documents_search import _return_individual_chunks
from app.models.common import DocumentSearchRequest


def _point(category: str, score: float, text: str = "текст чанка"):
    return qdrant_models.ScoredPoint(
        id=category,
        version=0,
        score=score,
        payload={
            "category_path": category,
            "raw_text": text,
            "source_id": "doc-1",
            "chunk_index": 0,
            "total_chunks": 1,
            "_collection_name": "test",
        },
    )


def _request(limit: int, max_text_length=None):
    return DocumentSearchRequest(
        query_text="запрос",
        limit=limit,
        max_text_length=max_text_length,
    )


@pytest.mark.asyncio
async def test_limit_is_applied_after_dedup():
    points = [_point(f"cat{i}", 1.0 - i / 10) for i in range(6)]

    results = await _return_individual_chunks(
        points, ["test"], _request(3), None
    )

    assert len(results) == 3
    # Порядок кандидатов сохраняется, лишние отбрасываются с конца
    assert [r.category_path for r in results] == ["cat0", "cat1", "cat2"]


@pytest.mark.asyncio
async def test_limit_equal_to_unique_count_keeps_all():
    points = [_point(f"cat{i}", 1.0) for i in range(6)]

    results = await _return_individual_chunks(
        points, ["test"], _request(6), None
    )

    assert len(results) == 6


@pytest.mark.asyncio
async def test_limit_larger_than_unique_count_does_not_pad():
    points = [_point(f"cat{i}", 1.0) for i in range(2)]

    results = await _return_individual_chunks(
        points, ["test"], _request(5), None
    )

    assert len(results) == 2


@pytest.mark.asyncio
async def test_max_text_length_still_applied_after_limit():
    points = [_point("cat0", 1.0, text="abcdef"), _point("cat1", 0.9)]

    results = await _return_individual_chunks(
        points, ["test"], _request(1, max_text_length=3), None
    )

    assert len(results) == 1
    assert results[0].document == "abc"
