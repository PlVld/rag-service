"""Тесты нормализации путей категорий в иерархии (эндпоинт /v1/admin/categories/hierarchy)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.api.admin import _normalize_category_path, _get_category_hierarchy_data


def test_normalize_category_path():
    # Канонический вид с " / " не меняется
    assert _normalize_category_path("Документация / API") == "Документация / API"
    assert _normalize_category_path("A / B / C") == "A / B / C"
    # Разделитель "/" без пробелов приводится к " / "
    assert _normalize_category_path("Документация/API") == "Документация / API"
    assert _normalize_category_path("A/B/C") == "A / B / C"
    assert _normalize_category_path(" A / B ") == "A / B"
    # Одиночная категория без разделителей
    assert _normalize_category_path("Docs") == "Docs"


async def test_hierarchy_normalizes_category_separator():
    """Пути с "/" без пробелов должны нормализоваться до " / " перед facet-запросом."""
    facet_calls = []

    async def mock_facet(collection_name, key, limit, facet_filter=None):
        facet_calls.append((key, facet_filter))
        return SimpleNamespace(hits=[])

    client = MagicMock()
    client.get_collections = AsyncMock(return_value=SimpleNamespace(
        collections=[SimpleNamespace(name="documents")]))
    client.facet = mock_facet

    with patch("app.api.health.get_client", return_value=client):
        await _get_category_hierarchy_data(
            collection_name="documents",
            depth=2,
            categories=["Документация/API"],
        )

    assert facet_calls, "facet не был вызван"
    for key, facet_filter in facet_calls:
        match_texts = [c.match.text for c in facet_filter.must if c.key == "category_path"]
        assert match_texts == ["Документация / API"], (
            f"Путь в фильтре не нормализован для {key}: {match_texts}"
        )


async def test_hierarchy_level_calculated_after_normalization():
    """Уровень пути должен считаться по нормализованному пути: "A/B" - это уровень 1."""
    facet_keys = []

    async def mock_facet(collection_name, key, limit, facet_filter=None):
        facet_keys.append(key)
        return SimpleNamespace(hits=[])

    client = MagicMock()
    client.get_collections = AsyncMock(return_value=SimpleNamespace(
        collections=[SimpleNamespace(name="documents")]))
    client.facet = mock_facet

    with patch("app.api.health.get_client", return_value=client):
        await _get_category_hierarchy_data(
            collection_name="documents",
            depth=2,
            categories=["Документация/API"],
        )

    # base_level=1 + depth=2 => уровни 1 и 2
    assert facet_keys == ["category_id_level1", "category_id_level2"]
