"""
Регрессионные тесты перемещения документов между категориями и серверного source_id.

Сценарии (спецификация):
- source_id всегда вычисляется сервером: gen(категория_поиска + имя_файла);
- new_category_path, отличная от category_path, перемещает найденный документ:
  старый source_id помечается неактуальным, документ создаётся под новым source_id,
  prev_source_ids накапливает цепочку перемещений, версия — сквозная;
- при любом содержимом (изменившемся и нет) — полная переобработка: категория входит
  в текст эмбеддинга, поэтому копировать чанки с векторами при смене категории нельзя;
- если по категории поиска документ не найден — файл создаётся под новой категорией;
- документ с тем же source_id, что у нового (цель), помечается is_latest=False
  c исключением новой версии (keep_version) и участвует в сквозной нумерации версий;
- lookup по хешу возвращает категории уровня документа.
"""
import hashlib
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models as qdrant_models
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.api import files as files_module
from app.repository.qdrant_repository import QdrantBatchWriter
from app.utils import generate_uuid_from_parts


COLLECTION = "test_docs"


def make_upload(filename: str, content: bytes) -> StarletteUploadFile:
    return StarletteUploadFile(io.BytesIO(content), filename=filename)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def make_point(source_id: str, doc_hash: str, version: int, is_latest: bool = True, **payload_extra):
    payload = {
        "source_id": source_id,
        "doc_hash": doc_hash,
        "version": version,
        "is_latest": is_latest,
        "original_filename": "doc.txt",
        "chunk_index": 0,
        "total_chunks": 1,
    }
    payload.update(payload_extra)
    return qdrant_models.PointStruct(id=generate_uuid_from_parts([source_id, version, 0]), vector=[0.1, 0.2], payload=payload)


class FakeQdrantClient:
    """Минимальный клиент: scroll с фильтрацией по must-условиям."""

    def __init__(self, points):
        self.points = points

    async def scroll(self, collection_name, scroll_filter=None, limit=10,
                     offset=None, with_payload=True, with_vectors=False):
        must = list(getattr(scroll_filter, "must", None) or [])
        result = []
        for p in self.points:
            if all(p.payload.get(c.key) == c.match.value for c in must):
                result.append(p)
        return result, None


# ---------------------------------------------------------------- pure helpers

def test_prepare_source_id_and_categories():
    sid, search_cat, new_cat = files_module._prepare_source_id_and_categories(
        '["Документация", "API"]', '["Руководства"]', "doc.txt"
    )
    assert sid == generate_uuid_from_parts(["Документация", "API", "doc.txt"])
    assert search_cat == ["Документация", "API"]
    assert new_cat == ["Руководства"]

    # Без категорий source_id выводится только из имени файла
    sid2, search2, new2 = files_module._prepare_source_id_and_categories(None, None, "doc.txt")
    assert sid2 == generate_uuid_from_parts(["doc.txt"])
    assert search2 == []
    assert new2 == []


def test_doc_category_path_from_payload():
    payload = {
        "category_path": "Документация / API / Глава 1",
        "doc_category_count": 2,
    }
    assert files_module._doc_category_path_from_payload(payload) == ["Документация", "API"]
    # doc_category_count отсутствует — берём весь путь
    assert files_module._doc_category_path_from_payload(
        {"category_path": ["A", "B"]}) == ["A", "B"]
    assert files_module._doc_category_path_from_payload({}) == []


# ---------------------------------------------------------------- move-flow

async def test_move_same_hash_reprocesses():
    """Перемещение неизменившегося файла: категория входит в текст эмбеддинга,
    поэтому чанки пересоздаются полностью (без копирования векторов)."""
    content = b"document content"
    doc_hash = sha256(content)
    old_sid = generate_uuid_from_parts(["Документация", "API", "doc.txt"])
    new_sid = generate_uuid_from_parts(["Руководства", "doc.txt"])

    old_point = make_point(
        old_sid, doc_hash, version=2,
        category_level0="Документация",
        category_level1="API",
    )
    client = FakeQdrantClient([old_point])

    async def fake_process_documents(request, batch_writer):
        doc = request.documents[0]
        return [doc.payload["source_id"]], ["pid-1", "pid-2"], []

    with patch("app.api.files.get_client", return_value=client), \
         patch("app.api.files.extract_text_from_file",
               new_callable=AsyncMock, return_value=("текст", "text")), \
         patch("app.api.files.process_documents", side_effect=fake_process_documents) as pd_mock:
        batch_writer = QdrantBatchWriter()
        result = await files_module.process_single_file(
            file=make_upload("doc.txt", content),
            collection_name=COLLECTION,
            source_format="text",
            batch_writer=batch_writer,
            category_path='["Документация", "API"]',
            new_category_path='["Руководства"]',
        )

    assert result["status"] == "moved"
    assert result["source_id"] == new_sid
    assert result["version"] == 3  # сквозная нумерация: старая версия 2 + 1
    assert result["prev_source_ids"] == [old_sid]
    assert result["chunk_ids"] == ["pid-1", "pid-2"]

    # Полная переобработка даже при совпадающем хеше
    pd_mock.assert_called_once()
    doc = pd_mock.call_args[0][0].documents[0]
    assert doc.payload["source_id"] == new_sid
    assert doc.payload["version"] == 3
    assert doc.payload["prev_source_ids"] == [old_sid]
    assert doc.payload["doc_hash"] == doc_hash
    assert doc.category_path == ["Руководства"]

    # Старый документ помечается неактуальным целиком (keep_version=None)
    markers = [(sid, keep) for sids in batch_writer._mark_not_latest.values() for sid, keep in sids]
    assert (old_sid, None) in markers
    # Новых точек через копирование не добавляется — только через process_documents
    assert COLLECTION not in batch_writer._points_by_collection


async def test_move_marks_target_versions_not_latest():
    """Баг-регресс: версии целевого source_id помечаются is_latest=False,
    но свежесозданная версия исключается (keep_version=new_version)
    и участвует в сквозной нумерации версий."""
    content = b"new content"
    doc_hash = sha256(content)
    old_sid = generate_uuid_from_parts(["Документация", "doc.txt"])
    new_sid = generate_uuid_from_parts(["Руководства", "doc.txt"])

    old_point = make_point(old_sid, sha256(b"old content"), version=2, category_level0="Документация")
    target_point = make_point(new_sid, sha256(b"target content"), version=4, category_level0="Руководства")
    client = FakeQdrantClient([old_point, target_point])

    async def fake_process_documents(request, batch_writer):
        doc = request.documents[0]
        return [doc.payload["source_id"]], ["pid-1"], []

    with patch("app.api.files.get_client", return_value=client), \
         patch("app.api.files.extract_text_from_file",
               new_callable=AsyncMock, return_value=("новый текст", "text")), \
         patch("app.api.files.process_documents", side_effect=fake_process_documents) as pd_mock:
        batch_writer = QdrantBatchWriter()
        result = await files_module.process_single_file(
            file=make_upload("doc.txt", content),
            collection_name=COLLECTION,
            source_format="text",
            batch_writer=batch_writer,
            category_path='["Документация"]',
            new_category_path='["Руководства"]',
        )

    assert result["status"] == "moved"
    assert result["source_id"] == new_sid
    # Сквозная нумерация: max(старая 2, целевая 4) + 1
    assert result["version"] == 5
    assert result["prev_source_ids"] == [old_sid]

    doc = pd_mock.call_args[0][0].documents[0]
    assert doc.payload["version"] == 5
    assert doc.payload["prev_source_ids"] == [old_sid]

    # Старый помечается целиком; цель — с исключением новой версии,
    # чтобы отложенный set_payload не задел свежесозданные точки
    markers = sorted(
        (sid, keep) for sids in batch_writer._mark_not_latest.values() for sid, keep in sids)
    assert markers == [(old_sid, None), (new_sid, 5)]


async def test_move_changed_hash_reprocesses():
    """Перемещение изменившегося файла: полная переобработка под новым source_id."""
    content = b"new content"
    doc_hash = sha256(content)
    old_sid = generate_uuid_from_parts(["Документация", "doc.txt"])
    new_sid = generate_uuid_from_parts(["Руководства", "doc.txt"])

    old_point = make_point(old_sid, sha256(b"old content"), version=1, category_level0="Документация")
    client = FakeQdrantClient([old_point])

    async def fake_process_documents(request, batch_writer):
        doc = request.documents[0]
        return [doc.payload["source_id"]], ["pid-1", "pid-2"], []

    with patch("app.api.files.get_client", return_value=client), \
         patch("app.api.files.extract_text_from_file",
               new_callable=AsyncMock, return_value=("новый текст", "text")), \
         patch("app.api.files.process_documents", side_effect=fake_process_documents) as pd_mock:
        batch_writer = QdrantBatchWriter()
        result = await files_module.process_single_file(
            file=make_upload("doc.txt", content),
            collection_name=COLLECTION,
            source_format="text",
            batch_writer=batch_writer,
            category_path='["Документация"]',
            new_category_path='["Руководства"]',
        )

    assert result["status"] == "moved"
    assert result["source_id"] == new_sid
    assert result["version"] == 2
    assert result["prev_source_ids"] == [old_sid]
    assert result["chunk_ids"] == ["pid-1", "pid-2"]

    # В process_documents ушёл новый source_id с категорией присвоения и цепочкой перемещений
    doc = pd_mock.call_args[0][0].documents[0]
    assert doc.payload["source_id"] == new_sid
    assert doc.payload["prev_source_ids"] == [old_sid]
    assert doc.category_path == ["Руководства"]

    markers = [(sid, keep) for sids in batch_writer._mark_not_latest.values() for sid, keep in sids]
    assert (old_sid, None) in markers


async def test_move_to_target_with_same_content_updates_in_place():
    """Перемещение туда, где тот же контент уже актуален: версия не создаётся."""
    content = b"shared content"
    doc_hash = sha256(content)
    old_sid = generate_uuid_from_parts(["Старая", "doc.txt"])
    target_sid = generate_uuid_from_parts(["Новая", "doc.txt"])

    old_point = make_point(old_sid, sha256(b"other"), version=5)
    target_point = make_point(target_sid, doc_hash, version=3, chunk_index=0)
    client = FakeQdrantClient([old_point, target_point])

    with patch("app.api.files.get_client", return_value=client), \
         patch("app.api.files.extract_text_from_file", new_callable=AsyncMock) as extract_mock:
        batch_writer = QdrantBatchWriter()
        result = await files_module.process_single_file(
            file=make_upload("doc.txt", content),
            collection_name=COLLECTION,
            source_format="text",
            batch_writer=batch_writer,
            category_path='["Старая"]',
            new_category_path='["Новая"]',
        )

    assert result["status"] == "moved"
    assert result["source_id"] == target_sid
    assert result["version"] == 3
    assert result["chunk_ids"] == [target_point.id]
    assert result["prev_source_ids"] == [old_sid]
    extract_mock.assert_not_awaited()  # контент не переизвлекается

    # Целевые точки перезаписаны с дописанной цепочкой перемещений
    (point,) = batch_writer._points_by_collection[COLLECTION]
    assert point.id == target_point.id
    assert point.payload["prev_source_ids"] == [old_sid]
    markers = [(sid, keep) for sids in batch_writer._mark_not_latest.values() for sid, keep in sids]
    assert markers == [(old_sid, None)]


async def test_new_category_without_existing_doc_creates_new():
    """Новая категория при ненайденном документе: файл создаётся под новой категорией."""
    content = b"brand new content"
    new_sid = generate_uuid_from_parts(["Новая", "doc.txt"])

    async def fake_process_documents(request, batch_writer):
        doc = request.documents[0]
        return [doc.payload["source_id"]], ["pid-1"], []

    with patch("app.api.files.get_client", return_value=FakeQdrantClient([])), \
         patch("app.api.files.extract_text_from_file",
               new_callable=AsyncMock, return_value=("текст", "text")), \
         patch("app.api.files.process_documents", side_effect=fake_process_documents) as pd_mock:
        batch_writer = QdrantBatchWriter()
        result = await files_module.process_single_file(
            file=make_upload("doc.txt", content),
            collection_name=COLLECTION,
            source_format="text",
            batch_writer=batch_writer,
            category_path='["Старая"]',
            new_category_path='["Новая"]',
        )

    assert result["status"] == "created"
    assert result["source_id"] == new_sid
    assert result["version"] == 1

    doc = pd_mock.call_args[0][0].documents[0]
    assert doc.payload["source_id"] == new_sid
    assert doc.category_path == ["Новая"]


async def test_extracted_markdown_not_reconverted():
    """Баг-регресс: Markdown, полученный от extractor (Docling), не должен повторно
    конвертироваться по клиентскому source_format='html'. Двойная конвертация
    через html2text схлопывала переводы строк, весь документ превращался в одну
    строку-заголовок, и category_level2 получал весь текст файла вместо заголовка."""
    content = b"html content"
    new_sid = generate_uuid_from_parts(["Категория", "doc.txt"])

    async def fake_process_documents(request, batch_writer):
        doc = request.documents[0]
        return [doc.payload["source_id"]], ["pid-1"], []

    with patch("app.api.files.get_client", return_value=FakeQdrantClient([])), \
         patch("app.api.files.extract_text_from_file",
               new_callable=AsyncMock,
               return_value=("# Заголовок\n\nТекст секции", "markdown")), \
         patch("app.api.files.process_documents", side_effect=fake_process_documents) as pd_mock:
        batch_writer = QdrantBatchWriter()
        result = await files_module.process_single_file(
            file=make_upload("doc.txt", content),
            collection_name=COLLECTION,
            source_format="html",
            batch_writer=batch_writer,
            category_path='["Категория"]',
        )

    assert result["status"] == "created"
    doc = pd_mock.call_args[0][0].documents[0]
    # Фактический формат текста — markdown (конвертация не нужна),
    # формат исходного файла клиента сохранён в original_format
    assert doc.payload["source_format"] == "markdown"
    assert doc.payload["original_format"] == "html"
    assert doc.text == "# Заголовок\n\nТекст секции"


async def test_same_category_new_category_ignored():
    """new_category_path совпадает с category_path — обычный путь, перемещения нет."""
    content = b"some content"
    doc_hash = sha256(content)
    old_sid = generate_uuid_from_parts(["Категория", "doc.txt"])

    old_point = make_point(old_sid, doc_hash, version=1)
    client = FakeQdrantClient([old_point])

    with patch("app.api.files.get_client", return_value=client):
        batch_writer = QdrantBatchWriter()
        result = await files_module.process_single_file(
            file=make_upload("doc.txt", content),
            collection_name=COLLECTION,
            source_format="text",
            batch_writer=batch_writer,
            category_path='["Категория"]',
            new_category_path='["Категория"]',
        )

    # Сценарий A: метаданные обновляются, версия не растёт, source_id прежний
    assert result["status"] == "updated"
    assert result["source_id"] == old_sid
    assert result["version"] == 1


# ---------------------------------------------------------------- lookup

async def test_lookup_by_hash():
    p1 = make_point(
        "sid-1", "hash-x", version=2,
        category_path="Документация / API / Глава 1",
        doc_category_count=2,
    )
    p1_dup = make_point("sid-1", "hash-x", version=2, chunk_index=1)
    p2 = make_point("sid-2", "hash-x", version=1, category_path=["Руководства", "Настройка"])
    client = FakeQdrantClient([p1, p1_dup, p2])

    with patch("app.api.files.get_client", return_value=client):
        response = await files_module.lookup_file(collection_name=COLLECTION, file=None, doc_hash="hash-x")

    data = response["data"]
    assert data["doc_hash"] == "hash-x"
    assert len(data["matches"]) == 2  # дедуп по source_id
    by_sid = {m["source_id"]: m for m in data["matches"]}
    assert by_sid["sid-1"]["category_path"] == ["Документация", "API"]  # без заголовков
    assert by_sid["sid-1"]["version"] == 2
    assert by_sid["sid-2"]["category_path"] == ["Руководства", "Настройка"]


async def test_lookup_requires_file_or_hash():
    with pytest.raises(Exception) as exc_info:
        await files_module.lookup_file(collection_name=COLLECTION, file=None, doc_hash=None)
    assert "Either 'file' or 'doc_hash'" in str(exc_info.value.detail)


# ---------------------------------------------------------------- batch writer

async def test_mark_not_latest_keep_version_none():
    """keep_version=None — помечаются все версии (must_not пуст)."""
    writer = QdrantBatchWriter()
    point = make_point("sid", "h", 1)
    writer.add_point(COLLECTION, point)
    writer.mark_old_versions_not_latest(COLLECTION, "sid", keep_version=None)

    fake_client = MagicMock()
    fake_client.get_collection = AsyncMock(return_value=object())
    captured = []

    async def capture_batch(collection_name, update_operations, wait):
        captured.extend(update_operations)

    fake_client.batch_update_points = capture_batch

    with patch("app.repository.qdrant_repository.get_client", return_value=fake_client):
        await writer.commit()

    set_payload_ops = [op for op in captured if isinstance(op, qdrant_models.SetPayloadOperation)]
    assert len(set_payload_ops) == 1
    assert set_payload_ops[0].set_payload.filter.must_not == []
    assert set_payload_ops[0].set_payload.payload == {"is_latest": False}


async def test_mark_not_latest_keep_version_excluded():
    """keep_version задан — версия исключается через must_not."""
    writer = QdrantBatchWriter()
    point = make_point("sid", "h", 1)
    writer.add_point(COLLECTION, point)
    writer.mark_old_versions_not_latest(COLLECTION, "sid", keep_version=3)

    fake_client = MagicMock()
    fake_client.get_collection = AsyncMock(return_value=object())
    captured = []

    async def capture_batch(collection_name, update_operations, wait):
        captured.extend(update_operations)

    fake_client.batch_update_points = capture_batch

    with patch("app.repository.qdrant_repository.get_client", return_value=fake_client):
        await writer.commit()

    set_payload_ops = [op for op in captured if isinstance(op, qdrant_models.SetPayloadOperation)]
    assert len(set_payload_ops) == 1
    must_not = set_payload_ops[0].set_payload.filter.must_not
    assert len(must_not) == 1
    assert must_not[0].key == "version"
    assert must_not[0].match.value == 3
