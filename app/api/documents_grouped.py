"""
Детерминированная выборка сгруппированного текста чанков.

В отличие от /v1/documents/search здесь нет семантического поиска: чанки
отбираются по коллекции, категориям и идентификаторам документов, а затем
группируются по паре «документ + категория». Внутри группы текст склеивается
в порядке chunk_index.

Выборка выполняется в две фазы: сначала скроллируется только метаинформация
(без raw_text), затем полный текст читается лишь для тех групп, которые попали
в ответ после применения limit. Это важно для больших коллекций: raw_text
составляет основную часть передаваемых данных.
"""
import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Depends
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.api.documents_utils import _normalize_category_path
from app.api.health import get_client
from app.auth import verify_api_key
from app.core.config import settings
from app.models.common import GroupedTextRequest
from app.openapi_md.loader import load_openapi_md
from app.utils import create_response

router = APIRouter(
    prefix="/v1/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)],
)
logger = logging.getLogger(__name__)

# Размер страницы scroll. Крупная страница сокращает число round-trip'ов
# к Qdrant; объём передаваемых данных всё равно определяется выборкой.
_SCROLL_PAGE_SIZE = 5000

# Минимальный набор полей для первой фазы: группировка и метаданные ответа.
_METADATA_PAYLOAD = [
    "source_id",
    "version",
    "category_path",
    "chunk_index",
    "total_chunks",
]

# Полный набор полей для второй фазы — только для выбранных групп.
_TEXT_PAYLOAD = _METADATA_PAYLOAD + ["original_filename", "raw_text"]


def _category_segments(category: str) -> List[str]:
    """Разбивает путь категории на уровни.

    В базе разделитель — " / ". Пользователь может передать "/" без пробелов,
    поэтому поддерживаются оба варианта.
    """
    if " / " in category:
        return [part.strip() for part in category.split(" / ") if part.strip()]
    return [part.strip() for part in category.split("/") if part.strip()]


def _normalize_category(category: str) -> str:
    """Приводит путь категории к виду, в котором он хранится в payload."""
    return " / ".join(_category_segments(category))


def _matches_category(payload_category: str, categories: List[str]) -> bool:
    """Проверяет точное совпадение категории или её вложенность."""
    for category in categories:
        if payload_category == category or payload_category.startswith(category + " / "):
            return True
    return False


def _category_filter_conditions(category: str) -> List[qdrant_models.Condition]:
    """Условия Qdrant для категории и всех её вложенных категорий."""
    segments = _category_segments(category)
    if not segments:
        return []

    # category_levelN индексируются как keyword. Совпадение всех уровней
    # означает, что точка относится к этой категории или лежит глубже.
    conditions: List[qdrant_models.Condition] = [
        qdrant_models.FieldCondition(
            key=f"category_level{level}",
            match=qdrant_models.MatchValue(value=segment),
        )
        for level, segment in enumerate(segments)
    ]

    # Страховка для точек, у которых category_levelN отсутствует: точное
    # совпадение по полному пути.
    conditions.append(
        qdrant_models.FieldCondition(
            key="category_path",
            match=qdrant_models.MatchValue(value=category),
        )
    )
    return conditions


def _category_should_conditions(
    normalized_categories: List[str],
) -> Optional[List[qdrant_models.Condition]]:
    """OR-условия по категориям для серверного фильтра Qdrant."""
    should: List[qdrant_models.Condition] = []
    for category in normalized_categories:
        conditions = _category_filter_conditions(category)
        if conditions:
            should.append(qdrant_models.Filter(must=conditions))
    return should or None


def _build_filter(
    request: GroupedTextRequest,
    normalized_categories: List[str],
) -> Optional[qdrant_models.Filter]:
    """Строит серверный фильтр Qdrant по версии, документам и категориям."""
    must: List[qdrant_models.Condition] = []

    if not request.include_old_versions:
        must.append(
            qdrant_models.FieldCondition(
                key="is_latest",
                match=qdrant_models.MatchValue(value=True),
            )
        )

    documents = [doc for doc in (request.documents or []) if doc]
    if documents:
        must.append(
            qdrant_models.FieldCondition(
                key="source_id",
                match=qdrant_models.MatchAny(any=documents),
            )
        )

    should = _category_should_conditions(normalized_categories)

    if not must and not should:
        return None
    return qdrant_models.Filter(must=must or None, should=should)


def _build_text_filter(
    request: GroupedTextRequest,
    normalized_categories: List[str],
    source_ids: Set[str],
) -> qdrant_models.Filter:
    """Фильтр второй фазы: только выбранные документы."""
    must: List[qdrant_models.Condition] = []

    if not request.include_old_versions:
        must.append(
            qdrant_models.FieldCondition(
                key="is_latest",
                match=qdrant_models.MatchValue(value=True),
            )
        )
    must.append(
        qdrant_models.FieldCondition(
            key="source_id",
            match=qdrant_models.MatchAny(any=sorted(source_ids)),
        )
    )

    return qdrant_models.Filter(
        must=must,
        should=_category_should_conditions(normalized_categories),
    )


async def _resolve_collections(client, collection_name: Optional[str]) -> List[str]:
    """Возвращает список коллекций для выборки."""
    if collection_name:
        return [collection_name]

    collections_response = await client.get_collections()
    return [
        collection.name
        for collection in collections_response.collections
        if collection.name != settings.category_collection
    ]


async def _scroll(
    client,
    collection_name: str,
    scroll_filter: Optional[qdrant_models.Filter],
    payload_fields: List[str],
):
    """Постранично скроллирует коллекцию, не накапливая все точки в памяти."""
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=payload_fields,
            with_vectors=False,
        )
        yield points
        if offset is None:
            break


def _group_key(collection_name: str, payload: Dict[str, Any]) -> Optional[Tuple[str, str, Any, str]]:
    """Ключ группы «документ + версия + категория»."""
    source_id = payload.get("source_id")
    if not source_id:
        return None
    return (
        collection_name,
        source_id,
        payload.get("version", 1),
        _normalize_category_path(payload.get("category_path")),
    )


async def _collect_metadata(
    client,
    collection_name: str,
    scroll_filter: Optional[qdrant_models.Filter],
    normalized_categories: List[str],
    strict: bool,
) -> Dict[Tuple[str, str, Any, str], Dict[str, Any]]:
    """Первая фаза: группировка по метаданным, без чтения raw_text."""
    groups: Dict[Tuple[str, str, Any, str], Dict[str, Any]] = {}

    try:
        async for points in _scroll(client, collection_name, scroll_filter, _METADATA_PAYLOAD):
            for point in points:
                payload = point.payload or {}
                key = _group_key(collection_name, payload)
                if key is None:
                    continue

                # Основной фильтр выполнен в Qdrant; проверка в Python —
                # страховка для нестандартных payload.
                if normalized_categories and not _matches_category(key[3], normalized_categories):
                    continue

                group = groups.setdefault(
                    key,
                    {"chunk_indices": [], "total_chunks": None},
                )
                group["chunk_indices"].append(payload.get("chunk_index", 0))
                if group["total_chunks"] is None and payload.get("total_chunks") is not None:
                    group["total_chunks"] = payload.get("total_chunks")
    except UnexpectedResponse as e:
        if "Not found: Collection" in str(e):
            if strict:
                raise ValueError(f"Collection '{collection_name}' not found") from e
            logger.warning(f"Collection '{collection_name}' not found, skipping")
            return {}
        raise

    return groups


async def _collect_texts(
    client,
    collection_name: str,
    scroll_filter: qdrant_models.Filter,
    selected_keys: Set[Tuple[str, str, Any, str]],
) -> Dict[Tuple[str, str, Any, str], List[Dict[str, Any]]]:
    """Вторая фаза: полный текст только по выбранным группам."""
    groups: Dict[Tuple[str, str, Any, str], List[Dict[str, Any]]] = {}

    async for points in _scroll(client, collection_name, scroll_filter, _TEXT_PAYLOAD):
        for point in points:
            payload = point.payload or {}
            key = _group_key(collection_name, payload)
            if key is None or key not in selected_keys:
                continue
            groups.setdefault(key, []).append(payload)

    return groups


async def get_grouped_text(
    collection_name: Optional[str] = None,
    categories: Optional[List[str]] = None,
    documents: Optional[List[str]] = None,
    include_old_versions: bool = False,
    max_text_length: int = 0,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Возвращает сгруппированный текст чанков без семантического поиска.

    Группировка выполняется по (коллекция, source_id, версия, category_path).
    Внутри группы чанки сортируются по chunk_index и склеиваются через перевод
    строки. Фильтр по категориям включает вложенные категории.

    Raises:
        ValueError: если явно указанная коллекция не найдена.
    """
    client = get_client()
    request = GroupedTextRequest(
        collection_name=collection_name,
        categories=categories,
        documents=documents,
        include_old_versions=include_old_versions,
        max_text_length=max_text_length,
        limit=limit,
    )

    normalized_categories = [
        _normalize_category(category)
        for category in (categories or [])
        if category and str(category).strip()
    ]
    base_filter = _build_filter(request, normalized_categories)
    collection_names = await _resolve_collections(client, request.collection_name)

    strict = bool(request.collection_name)
    if len(collection_names) == 1:
        collected = [
            await _collect_metadata(
                client, collection_names[0], base_filter, normalized_categories, strict
            )
        ]
    else:
        # Коллекции независимы — скроллим их параллельно.
        collected = await asyncio.gather(*[
            _collect_metadata(client, name, base_filter, normalized_categories, False)
            for name in collection_names
        ])

    groups: Dict[Tuple[str, str, Any, str], Dict[str, Any]] = {}
    for collection_groups in collected:
        groups.update(collection_groups)

    ordered_keys = sorted(
        groups,
        key=lambda key: (
            key[0],
            key[1],
            min(groups[key]["chunk_indices"]) if groups[key]["chunk_indices"] else 0,
            key[3] or "",
        ),
    )
    selected_keys = set(ordered_keys[:limit])
    if not selected_keys:
        return []

    # Вторая фаза — только для групп, попавших в ответ.
    selected_by_collection: Dict[str, Set[str]] = defaultdict(set)
    for col_name, source_id, _version, _category_path in selected_keys:
        selected_by_collection[col_name].add(source_id)

    text_results = await asyncio.gather(*[
        _collect_texts(
            client,
            col_name,
            _build_text_filter(request, normalized_categories, source_ids),
            selected_keys,
        )
        for col_name, source_ids in selected_by_collection.items()
    ])

    texts: Dict[Tuple[str, str, Any, str], List[Dict[str, Any]]] = {}
    for collection_texts in text_results:
        texts.update(collection_texts)

    results: List[Dict[str, Any]] = []
    for key in ordered_keys:
        if key not in selected_keys:
            continue

        col_name, source_id, version, category_path = key
        meta = groups[key]
        payloads = sorted(
            texts.get(key, []),
            key=lambda item: item.get("chunk_index", 0),
        )

        text = "\n".join(item.get("raw_text", "") for item in payloads)
        if max_text_length > 0:
            text = text[:max_text_length]

        chunk_indices = sorted(meta["chunk_indices"])
        total_chunks = meta["total_chunks"]
        if total_chunks is None:
            total_chunks = chunk_indices[-1] + 1 if chunk_indices else 0

        results.append({
            "source_id": source_id,
            "collection_name": col_name,
            "category_path": category_path or None,
            "version": version,
            "first_chunk_index": chunk_indices[0] if chunk_indices else None,
            "last_chunk_index": chunk_indices[-1] if chunk_indices else None,
            "chunk_indices": chunk_indices,
            "chunk_count": len(chunk_indices),
            "total_chunks": total_chunks,
            "original_filename": payloads[0].get("original_filename") if payloads else None,
            "document": text,
        })

    return results


@router.post(
    "/grouped-text",
    summary="Сгруппированный текст чанков",
    description=load_openapi_md("documents_grouped_text.md"),
)
async def grouped_text(request: GroupedTextRequest):
    """
    Возвращает текст чанков, сгруппированный по документу и категории.

    В отличие от /search не использует семантический поиск: отбор выполняется
    по коллекции, категориям и списку документов (source_id).
    """
    try:
        results = await get_grouped_text(
            collection_name=request.collection_name,
            categories=request.categories,
            documents=request.documents,
            include_old_versions=request.include_old_versions,
            max_text_length=request.max_text_length,
            limit=request.limit,
        )
        return create_response(success=True, data={"results": results})
    except ValueError as e:
        return create_response(
            success=False,
            error_code="collection_not_found",
            error_message=str(e),
        )
    except Exception as e:
        logger.exception("Grouped text retrieval failed")
        return create_response(
            success=False,
            error_code="grouped_text_failed",
            error_message=str(e),
        )
