"""Тесты актуализации коллекции categories по коллекции-источнику."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from app.api.categories import (
    _build_category_nodes,
    sync_categories_from_collection,
)
from app.core.config import settings
from app.utils import generate_uuid_from_parts


def _point(point_id, payload):
    point = MagicMock()
    point.id = point_id
    point.payload = payload
    return point


def _not_found(name):
    return UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        content=f"Not found: Collection `{name}`".encode("utf-8"),
        headers={},
    )


async def _fake_encode(texts, task_type="document"):
    return [[0.1, 0.2, 0.3] for _ in texts]


def _page(*payloads):
    """Страница scroll: точки с payload и отсутствие следующего offset."""
    return ([_point(f"src-{i}", payload) for i, payload in enumerate(payloads)], None)


def _make_client(
    *,
    source_exists=True,
    categories_exists=True,
    has_is_latest=False,
    existing_payloads=None,
    scroll_pages=None,
):
    """Собирает мок AsyncQdrantClient с маршрутизацией по имени коллекции."""
    client = MagicMock()

    async def get_collection(name):
        if name == "source":
            if not source_exists:
                raise _not_found(name)
            info = MagicMock()
            info.payload_schema = {"is_latest": MagicMock()} if has_is_latest else {}
            return info
        if name == settings.category_collection:
            if not categories_exists:
                raise _not_found(name)
            return MagicMock()
        raise AssertionError(f"Unexpected collection: {name}")

    client.get_collection = AsyncMock(side_effect=get_collection)

    existing_payloads = existing_payloads or {}

    async def retrieve(collection_name, ids, with_payload=True, with_vectors=False):
        if not categories_exists:
            raise _not_found(collection_name)
        return [
            _point(point_id, existing_payloads[point_id])
            for point_id in ids
            if point_id in existing_payloads
        ]

    client.retrieve = AsyncMock(side_effect=retrieve)

    pages = list(scroll_pages or [])

    async def scroll(**kwargs):
        if not pages:
            return [], None
        return pages.pop(0)

    client.scroll = AsyncMock(side_effect=scroll)
    client.upsert = AsyncMock()
    client.batch_update_points = AsyncMock()
    return client


class TestCategorySync:
    @pytest.mark.asyncio
    async def test_creates_missing_nodes_with_parents(self):
        client = _make_client(
            scroll_pages=[_page(
                {
                    "category_path": "A / B / C",
                    "category_level0": "A",
                    "category_level1": "B",
                    "category_level2": "C",
                },
                {
                    "category_path": "A / D",
                    "category_level0": "A",
                    "category_level1": "D",
                },
            )],
        )

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ) as encode:
            result = await sync_categories_from_collection("source")

        assert result["scan_mode"] == "scroll"
        assert result["source_category_paths"] == 2
        assert result["category_nodes"] == 4
        assert result["parent_nodes"] == 2
        assert result["missing"] == 4
        assert result["outdated"] == 0
        assert result["unchanged"] == 0
        assert result["created"] == 4
        assert result["updated"] == 0
        assert result["embedded"] == 4

        encode.assert_awaited_once()
        client.upsert.assert_awaited_once()
        points = client.upsert.await_args.kwargs["points"]
        assert len(points) == 4

        by_path = {point.payload["category_path"]: point for point in points}
        assert by_path["A"].payload["parent_id"] is None
        assert by_path["A / B"].payload["parent_id"] == generate_uuid_from_parts(["A"])
        assert by_path["A / B / C"].payload["category_id"] == generate_uuid_from_parts(
            ["A / B / C"]
        )
        assert by_path["A / B / C"].payload["category_level2"] == "C"
        assert by_path["A / B / C"].payload["category_id_level0"] == generate_uuid_from_parts(
            ["A"]
        )

    @pytest.mark.asyncio
    async def test_internal_separator_in_level_name_is_preserved(self):
        """Название уровня может содержать " / " — его нельзя дробить."""
        client = _make_client(
            scroll_pages=[_page({
                "category_path": "A / B (X / Y)",
                "category_level0": "A",
                "category_level1": "B (X / Y)",
            })],
        )

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ):
            result = await sync_categories_from_collection("source")

        assert result["source_category_paths"] == 1
        assert result["category_nodes"] == 2

        points = client.upsert.await_args.kwargs["points"]
        by_path = {point.payload["category_path"]: point for point in points}
        assert set(by_path) == {"A", "A / B (X / Y)"}
        leaf = by_path["A / B (X / Y)"].payload
        assert leaf["category_level"] == 1
        assert leaf["category_level1"] == "B (X / Y)"
        assert leaf["category_name"] == "B (X / Y)"
        assert leaf["parent_id"] == generate_uuid_from_parts(["A"])
        assert "A / B (X" not in by_path

    @pytest.mark.asyncio
    async def test_falls_back_to_category_path_without_levels(self):
        client = _make_client(
            scroll_pages=[_page({"category_path": "A / B"})],
        )

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ):
            result = await sync_categories_from_collection("source")

        assert result["category_nodes"] == 2
        points = client.upsert.await_args.kwargs["points"]
        assert {point.payload["category_path"] for point in points} == {"A", "A / B"}

    @pytest.mark.asyncio
    async def test_idempotent_when_nodes_are_up_to_date(self):
        payloads = [
            {
                "category_path": "A / B / C",
                "category_level0": "A",
                "category_level1": "B",
                "category_level2": "C",
            },
            {
                "category_path": "A / D",
                "category_level0": "A",
                "category_level1": "D",
            },
        ]
        nodes = _build_category_nodes([["A", "B", "C"], ["A", "D"]])
        existing = {node["id"]: node["payload"] for node in nodes.values()}
        client = _make_client(
            scroll_pages=[_page(*payloads)],
            existing_payloads=existing,
        )

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ) as encode:
            result = await sync_categories_from_collection("source")

        assert result["missing"] == 0
        assert result["outdated"] == 0
        assert result["unchanged"] == 4
        assert result["created"] == 0
        assert result["updated"] == 0
        assert result["embedded"] == 0

        encode.assert_not_awaited()
        client.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_updates_outdated_payload_and_keeps_extra_fields(self):
        nodes = _build_category_nodes([["A", "B"]])
        existing = {node["id"]: dict(node["payload"]) for node in nodes.values()}
        outdated_id = generate_uuid_from_parts(["A / B"])
        existing[outdated_id].pop("category_id_level1")
        existing[outdated_id]["custom_field"] = "keep-me"
        client = _make_client(
            scroll_pages=[_page({
                "category_path": "A / B",
                "category_level0": "A",
                "category_level1": "B",
            })],
            existing_payloads=existing,
        )

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ) as encode:
            result = await sync_categories_from_collection("source")

        assert result["missing"] == 0
        assert result["outdated"] == 1
        assert result["unchanged"] == 1
        assert result["updated"] == 1
        assert result["embedded"] == 1

        encode.assert_awaited_once()
        points = client.upsert.await_args.kwargs["points"]
        assert len(points) == 1
        assert points[0].payload["custom_field"] == "keep-me"
        assert points[0].payload["category_id_level1"] == outdated_id

    @pytest.mark.asyncio
    async def test_dry_run_does_not_embed_or_write(self):
        client = _make_client(
            scroll_pages=[_page({
                "category_path": "A / B / C",
                "category_level0": "A",
                "category_level1": "B",
                "category_level2": "C",
            })],
        )

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ) as encode:
            result = await sync_categories_from_collection("source", dry_run=True)

        assert result["dry_run"] is True
        assert result["missing"] == 3
        assert result["created"] == 0
        assert result["updated"] == 0
        assert result["embedded"] == 0
        assert result["sample_missing"]

        encode.assert_not_awaited()
        client.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_source_collection(self):
        client = _make_client(source_exists=False)

        with patch("app.api.categories.get_client", return_value=client):
            with pytest.raises(ValueError, match="not found"):
                await sync_categories_from_collection("source")

    @pytest.mark.asyncio
    async def test_rejects_target_category_collection(self):
        client = _make_client()

        with patch("app.api.categories.get_client", return_value=client):
            with pytest.raises(ValueError, match="target category collection"):
                await sync_categories_from_collection(settings.category_collection)

    @pytest.mark.asyncio
    async def test_is_latest_filter(self):
        payload = {
            "category_path": "A",
            "category_level0": "A",
        }

        client = _make_client(
            scroll_pages=[_page(payload)],
            has_is_latest=True,
        )
        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ):
            await sync_categories_from_collection("source")

        scroll_filter = client.scroll.await_args.kwargs["scroll_filter"]
        assert scroll_filter is not None

        client_old = _make_client(
            scroll_pages=[_page(payload)],
            has_is_latest=True,
        )
        with patch("app.api.categories.get_client", return_value=client_old), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ):
            await sync_categories_from_collection(
                "source",
                include_old_versions=True,
            )

        assert client_old.scroll.await_args.kwargs["scroll_filter"] is None

    @pytest.mark.asyncio
    async def test_empty_source_collection(self):
        client = _make_client(scroll_pages=[])

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ) as encode:
            result = await sync_categories_from_collection("source")

        assert result["scan_mode"] == "scroll"
        assert result["source_category_paths"] == 0
        assert result["category_nodes"] == 0
        assert result["created"] == 0
        assert result["points_scanned"] == 0
        assert result["points_with_wrong_category_ids"] == 0
        encode.assert_not_awaited()
        client.upsert.assert_not_awaited()
        client.batch_update_points.assert_not_awaited()


class TestCategoryIdFix:
    """Исправление category_id_levelN в чанках по коллекции categories."""

    @pytest.mark.asyncio
    async def test_correct_ids_are_not_rewritten(self):
        nodes = _build_category_nodes([["A", "B"]])
        leaf = nodes["A / B"]["payload"]
        chunk_payload = {
            "category_path": "A / B",
            "category_level0": "A",
            "category_level1": "B",
            "category_id_level0": leaf["category_id_level0"],
            "category_id_level1": leaf["category_id"],
            "category_id": leaf["category_id"],
        }
        existing = {node["id"]: node["payload"] for node in nodes.values()}
        client = _make_client(
            scroll_pages=[_page(chunk_payload)],
            existing_payloads=existing,
        )

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ):
            result = await sync_categories_from_collection("source")

        assert result["points_scanned"] == 1
        assert result["points_with_wrong_category_ids"] == 0
        assert result["points_updated"] == 0
        assert result["sample_wrong_category_ids"] == []
        client.batch_update_points.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wrong_ids_are_replaced_with_deterministic_ones(self):
        old_root = "11111111-1111-1111-1111-111111111111"
        old_leaf = "22222222-2222-2222-2222-222222222222"
        chunk_payload = {
            "category_path": "A / B",
            "category_level0": "A",
            "category_level1": "B",
            "category_id_level0": old_root,
            "category_id_level1": old_leaf,
            "category_id": old_leaf,
        }
        client = _make_client(scroll_pages=[_page(chunk_payload)])

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ):
            result = await sync_categories_from_collection("source")

        assert result["points_with_wrong_category_ids"] == 1
        assert result["points_updated"] == 1
        assert result["sample_wrong_category_ids"] == ["A / B (1 точек)"]

        client.batch_update_points.assert_awaited_once()
        kwargs = client.batch_update_points.await_args.kwargs
        assert kwargs["collection_name"] == "source"
        assert kwargs["wait"] is True
        assert len(kwargs["update_operations"]) == 1

        operation = kwargs["update_operations"][0]
        expected_root = generate_uuid_from_parts(["A"])
        expected_leaf = generate_uuid_from_parts(["A / B"])
        assert operation.set_payload.payload == {
            "category_id_level0": expected_root,
            "category_id_level1": expected_leaf,
            "category_id": expected_leaf,
        }
        assert operation.set_payload.points == ["src-0"]

    @pytest.mark.asyncio
    async def test_category_id_is_not_added_when_absent(self):
        """В старых коллекциях поля category_id нет — не создаём его."""
        chunk_payload = {
            "category_path": "A",
            "category_level0": "A",
            "category_id_level0": "33333333-3333-3333-3333-333333333333",
        }
        client = _make_client(scroll_pages=[_page(chunk_payload)])

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ):
            await sync_categories_from_collection("source")

        operation = client.batch_update_points.await_args.kwargs["update_operations"][0]
        assert operation.set_payload.payload == {
            "category_id_level0": generate_uuid_from_parts(["A"]),
        }

    @pytest.mark.asyncio
    async def test_dry_run_reports_but_does_not_write_ids(self):
        chunk_payload = {
            "category_path": "A",
            "category_level0": "A",
            "category_id_level0": "44444444-4444-4444-4444-444444444444",
        }
        client = _make_client(scroll_pages=[_page(chunk_payload)])

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ):
            result = await sync_categories_from_collection("source", dry_run=True)

        assert result["points_with_wrong_category_ids"] == 1
        assert result["points_updated"] == 0
        client.batch_update_points.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_path_without_levels_is_not_touched(self):
        """Без category_levelN путь может содержать " / " в названии — не правим."""
        client = _make_client(
            scroll_pages=[_page({"category_path": "A / B (X / Y)"})],
        )

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ):
            result = await sync_categories_from_collection("source")

        assert result["points_with_wrong_category_ids"] == 0
        client.batch_update_points.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_id_fix_is_idempotent(self):
        node = _build_category_nodes([["A"]])["A"]
        chunk_payload = {
            "category_path": "A",
            "category_level0": "A",
            "category_id_level0": node["payload"]["category_id"],
        }
        client = _make_client(
            scroll_pages=[_page(chunk_payload)],
            existing_payloads={node["id"]: node["payload"]},
        )

        with patch("app.api.categories.get_client", return_value=client), \
                patch(
                    "app.api.categories.encode_text",
                    new=AsyncMock(side_effect=_fake_encode),
                ):
            result = await sync_categories_from_collection("source")

        assert result["points_with_wrong_category_ids"] == 0
        assert result["missing"] == 0
        assert result["outdated"] == 0
        client.batch_update_points.assert_not_awaited()

