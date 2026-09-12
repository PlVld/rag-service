"""
Регрессионные тесты для генерации и сохранения полей category_levelN / category_id_levelN.

Покрывают сценарий: обновление категорий документа не должно терять category_id_level*
и не должно оставлять протухшие ID уровней глубже нового пути.
"""
import pytest
from qdrant_client.http import models as qdrant_models

from app.utils import apply_category_levels, generate_uuid_from_parts
from app.api.files import _update_point_payload


def _make_point(payload):
    return qdrant_models.PointStruct(
        id="test-point-1",
        vector=[0.1, 0.2, 0.3],
        payload=payload,
    )


class TestApplyCategoryLevels:
    def test_generates_level_fields(self):
        payload = {}
        apply_category_levels(payload, ["Документация", "API"])
        assert payload["category_path"] == "Документация / API"
        assert payload["category_level"] == 1
        assert payload["category_level0"] == "Документация"
        assert payload["category_level1"] == "API"

    def test_id_levels_are_deterministic(self):
        payload = {}
        apply_category_levels(payload, ["A", "B", "C"])
        assert payload["category_id_level0"] == generate_uuid_from_parts(["A"])
        assert payload["category_id_level1"] == generate_uuid_from_parts(["A / B"])
        assert payload["category_id_level2"] == generate_uuid_from_parts(["A / B / C"])

    def test_removes_stale_deeper_levels(self):
        # Старый путь глубиной 3, новый — глубиной 2:
        # уровни 2-3 не должны остаться в payload
        payload = {
            "category_level0": "A", "category_level1": "B", "category_level2": "C",
            "category_id_level0": "id-a", "category_id_level1": "id-ab",
            "category_id_level2": "id-abc",
            "category_level": 2,
        }
        apply_category_levels(payload, ["X", "Y"])
        assert "category_level2" not in payload
        assert "category_id_level2" not in payload
        assert payload["category_level"] == 1
        assert payload["category_level0"] == "X"
        assert payload["category_level1"] == "Y"
        assert payload["category_id_level0"] == generate_uuid_from_parts(["X"])

    def test_empty_list_preserves_existing_fields(self):
        payload = {"category_level0": "A", "category_id_level0": "id-a"}
        apply_category_levels(payload, [])
        assert payload == {"category_level0": "A", "category_id_level0": "id-a"}


class TestUpdatePointPayload:
    @pytest.mark.asyncio
    async def test_category_update_regenerates_id_levels(self):
        # Регрессия: раньше _update_point_payload удалял category_id_level*
        # при обновлении категорий и не генерировал их заново
        point = _make_point({
            "raw_text": "текст",
            "category_path": "Старая / Категория",
            "category_level0": "Старая",
            "category_level1": "Категория",
            "category_id_level0": "old-id-0",
            "category_id_level1": "old-id-1",
            "category_level": 1,
        })

        updated = await _update_point_payload(
            point=point,
            file_path="/docs/new.md",
            category_list=["Новая", "Категория"],
            source_format="markdown",
            original_filename="new.md",
            doc_hash="abc123",
        )

        payload = updated.payload
        assert payload["category_path"] == "Новая / Категория"
        assert payload["category_level0"] == "Новая"
        assert payload["category_level1"] == "Категория"
        # ID перегенерированы из нового пути
        assert payload["category_id_level0"] == generate_uuid_from_parts(["Новая"])
        assert payload["category_id_level1"] == generate_uuid_from_parts(["Новая / Категория"])
        assert payload["category_level"] == 1

    @pytest.mark.asyncio
    async def test_shorter_path_removes_deeper_id_levels(self):
        # Регрессия: при смене пути на более короткий старые глубокие
        # category_id_level* не должны оставаться в payload
        point = _make_point({
            "raw_text": "текст",
            "category_path": "A / B / C",
            "category_level0": "A",
            "category_level1": "B",
            "category_level2": "C",
            "category_id_level0": "id-0",
            "category_id_level1": "id-1",
            "category_id_level2": "id-2",
            "category_level": 2,
        })

        updated = await _update_point_payload(
            point=point,
            file_path="/docs/new.md",
            category_list=["X", "Y"],
            source_format="markdown",
            original_filename="new.md",
            doc_hash="abc123",
        )

        payload = updated.payload
        assert "category_level2" not in payload
        assert "category_id_level2" not in payload
        assert payload["category_level"] == 1

    @pytest.mark.asyncio
    async def test_no_categories_preserves_existing_levels(self):
        # Без новых категорий существующие поля категорий не трогаем
        point = _make_point({
            "raw_text": "текст",
            "category_path": "A / B",
            "category_level0": "A",
            "category_level1": "B",
            "category_id_level0": "id-0",
            "category_id_level1": "id-1",
            "category_level": 1,
        })

        updated = await _update_point_payload(
            point=point,
            file_path="/docs/new.md",
            category_list=[],
            source_format="markdown",
            original_filename="new.md",
            doc_hash="abc123",
        )

        payload = updated.payload
        assert payload["category_path"] == "A / B"
        assert payload["category_id_level0"] == "id-0"
        assert payload["category_id_level1"] == "id-1"
        assert payload["category_level"] == 1
