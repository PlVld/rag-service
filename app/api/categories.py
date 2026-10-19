import logging
import time
from typing import List, Optional, Dict, Tuple, Any, Literal
from fastapi import APIRouter, Query, Depends
from app.auth import verify_api_key
from app.openapi_md.loader import load_openapi_md

from pydantic import BaseModel, Field, model_serializer, field_validator

from app.api.health import get_client
from app.core.config import settings
from app.core.exact_search import matched_phrases, normalize_phrases
from app.repository.qdrant_repository import QdrantBatchWriter
from app.utils import basic_normalize, generate_uuid_from_parts, scroll_all_pages
from app.core.embeddings import encode_text
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

router = APIRouter(prefix="/v1/categories",
                   tags=["Categories"],
                   dependencies=[Depends(verify_api_key)],)
logger = logging.getLogger(__name__)


class CategorySearchRequest(BaseModel):
    query_text: str = Field(..., description="Текст запроса для поиска по категориям", examples=["Документация по API"])
    limit: int = Field(10, description="Максимальное количество результатов (1-100)", ge=1, le=100, examples=[10])
    min_score: Optional[float] = Field(
        default=settings.search_min_score,
        description=(
            "Минимальный score категории (0..1). Категории с меньшим "
            "косинусным сходством отбрасываются. По умолчанию 0.55; "
            "null - тоже значение по умолчанию, 0 - без ограничения."
        ),
        ge=0.0,
        le=1.0,
        examples=[0.55, None]
    )
    grouped: bool = Field(False, description="Если True, возвращает категории, сгруппированные по коллекциям", examples=[False])
    exact_phrases: Optional[List[str]] = Field(
        default=None,
        max_length=10,
        description=(
            "Ключевые слова или фразы для точного поиска по путям категорий. "
            "Совпадения добавляются к семантическим результатам и идут первыми."
        ),
        examples=[["Документация", "API"], None]
    )
    phrase_mode: Literal["any", "all"] = Field(
        default="any",
        description=(
            "Режим exact-поиска: any - достаточно любой фразы, all - должны "
            "встретиться все фразы."
        ),
        examples=["any", "all"]
    )
    fields: Optional[List[str]] = Field(
        default=None,
        description=(
            "Список полей для возврата. Если не указан, возвращаются только `path` и `score`. "
            "Доступные поля: `id`, `score`, `category_name`, `category_path`, `categories`, "
            "`category_level`, `category_id`, `levels`, `id_levels`, "
            "`exact_match`, `matched_phrases`."
        ),
        examples=[["path", "score"]]
    )

    @field_validator("min_score", mode="before")
    @classmethod
    def default_min_score(cls, value):
        """null от клиента означает «значение по умолчанию», а не отключение."""
        return settings.search_min_score if value is None else value


class CategoryResponse(BaseModel):
    id: str
    score: float
    category_name: str
    category_path: str
    categories: List[str]
    category_level: int
    category_id: Optional[str] = None
    levels: Dict[str, str] = Field(default_factory=dict)
    id_levels: Dict[str, str] = Field(default_factory=dict)
    exact_match: Optional[bool] = Field(
        default=None,
        description="Найдена ли категория по точной фразе (exact_phrases)"
    )
    matched_phrases: Optional[List[str]] = Field(
        default=None,
        description="Какие из exact_phrases встретились в пути категории"
    )

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        """Убирает служебные поля exact-поиска, когда они не заполнены."""
        data = handler(self)
        for field in ("exact_match", "matched_phrases"):
            if data.get(field) is None:
                data.pop(field, None)
        return data

class CategoriesSearchResponse(BaseModel):
    results: List[CategoryResponse]


def _build_level_payload(path: List[str]) -> Dict[str, str]:
    return {f"category_level{i}": path[i] for i in range(len(path))}


async def update_category_hierarchy(path: List[str]) -> Tuple[List[str], Dict[str, str], List[qdrant_models.PointStruct]]:
    """
    Обновляет иерархию категорий в векторной базе данных.

    Возвращает:
        category_ids: список UUID для всех уровней иерархии
        level_payload: словарь с уровнями названий (category_level0, category_level1, ...)
        points_to_upsert: список точек Qdrant для сохранения
    """
    if not path:
        return [], {}, []

    client = get_client()
    full_paths = [" / ".join(path[: i + 1]) for i in range(len(path))]
    category_ids = [generate_uuid_from_parts([fp]) for fp in full_paths]

    existing_dict = {}
    try:
        existing_points = await client.retrieve(
            collection_name=settings.category_collection,
            ids=category_ids,
            with_payload=True
        )
        existing_dict = {point.id: point.payload for point in existing_points}
    except UnexpectedResponse as e:
        if "Not found: Collection" not in str(e):
            raise

    points_to_upsert = []
    parent_id = None
    paths_to_encode = []
    point_data = []
    
    for i, name in enumerate(path):
        cat_id = category_ids[i]
        full_path = full_paths[i]
        level_payload = _build_level_payload(path[: i + 1])
        id_level_payload = {f"category_id_level{j}": category_ids[j] for j in range(i + 1)}

        expected_payload = {
            "category_name": name,
            "category_path": full_path,
            "category_level": i,
            "parent_id": parent_id,
            "category_id": cat_id,
            **level_payload,
            **id_level_payload,
        }
        
        if cat_id not in existing_dict:
            paths_to_encode.append(full_path)
            point_data.append((cat_id, expected_payload))
        else:
            existing = existing_dict[cat_id]
            need_update = False
            
            if (existing.get("category_name") in (None, "") or
                    existing.get("category_id") in (None, "") or
                    "parent_id" not in existing):
                need_update = True
            elif (existing.get("parent_id") != parent_id or
                  not all(f"category_id_level{j}" in existing for j in range(i + 1))):
                need_update = True

            if need_update:
                paths_to_encode.append(full_path)
                point_data.append((cat_id, expected_payload))
        
        parent_id = cat_id

    if paths_to_encode:
        # encode_text для списка строк возвращает List[List[float]]
        vectors: List[List[float]] = await encode_text(paths_to_encode, task_type="document")  # type: ignore
        
        for (cat_id, payload), vector in zip(point_data, vectors):
            points_to_upsert.append(
                qdrant_models.PointStruct(id=cat_id, vector=vector, payload=payload)
            )

    final_levels = _build_level_payload(path)
    return category_ids, final_levels, points_to_upsert


# --- Актуализация коллекции categories по данным другой коллекции ---
#
# Категории в Qdrant хранятся отдельно от документов. Если коллекция документов
# перенесена с другой машины, её точки содержат category_path и category_id_levelN,
# но узлов в коллекции categories может не быть. Эта операция читает уникальные
# пути из коллекции-источника, достраивает всех родителей и идемпотентно
# создаёт/исправляет узлы категорий.

_CATEGORY_SCAN_BATCH = 1000
_CATEGORY_RETRIEVE_BATCH = 1000
_CATEGORY_UPSERT_BATCH = 256
# Пачки для исправления category_id_levelN в чанках коллекции-источника.
_CATEGORY_SET_PAYLOAD_BATCH = 500
_CATEGORY_BATCH_UPDATE_BATCH = 500
# Ограничение только для восстановления уровней из payload, защита от
# бесконечного цикла на повреждённых данных.
_CATEGORY_MAX_RECONSTRUCT_LEVELS = 32
_CATEGORY_PREVIEW_LIMIT = 20


def _category_ids_for_parts(parts: List[str]) -> List[str]:
    """Детерминированные ID узлов для всех префиксов пути."""
    return [
        generate_uuid_from_parts([" / ".join(parts[: level + 1])])
        for level in range(len(parts))
    ]


def _has_explicit_category_levels(payload: Dict[str, Any]) -> bool:
    """Есть ли в payload исходные уровни (``category_level0``).

    Без них ``category_path`` неоднозначен: название уровня может содержать
    разделитель. Исправлять ID в таких точках небезопасно.
    """
    return payload.get("category_level0") not in (None, "")



def _split_category_path(value: Any) -> List[str]:
    """Разбирает путь категории из строки или массива в список уровней.

    Используется только как fallback, когда в payload нет category_levelN:
    разделитель " / " неоднозначен, если название уровня само его содержит.
    """
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value]
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if " / " in text:
            parts = [part.strip() for part in text.split(" / ")]
        else:
            parts = [part.strip() for part in text.split("/")]
    else:
        return []
    return [part for part in parts if part]


def _payload_category_parts(payload: Dict[str, Any]) -> List[str]:
    """Возвращает уровни категории из payload.

    Приоритет у ``category_levelN``: это исходные уровни, в которых название
    может содержать " / " (например, "Как работает ЕСТЬNULL (ISNULL / COALESCE)").
    Разбор ``category_path`` применяется только если уровней нет.
    """
    parts = []
    for level in range(_CATEGORY_MAX_RECONSTRUCT_LEVELS):
        value = payload.get(f"category_level{level}")
        if value in (None, ""):
            break
        parts.append(str(value).strip())
    if parts:
        return [part for part in parts if part]

    return _split_category_path(payload.get("category_path"))


def _build_expected_category_payload(path: List[str]) -> Tuple[str, Dict[str, Any]]:
    """Строит детерминированный ID и payload узла категории.

    Правила совпадают с ``update_category_hierarchy``: ID узла — UUID5 от
    полного пути, parent_id — ID родительского префикса.
    """
    category_ids = _category_ids_for_parts(path)
    full_path = " / ".join(path)
    payload: Dict[str, Any] = {
        "category_name": path[-1],
        "category_path": full_path,
        "category_level": len(path) - 1,
        "parent_id": category_ids[-2] if len(category_ids) > 1 else None,
        "category_id": category_ids[-1],
    }
    for level, name in enumerate(path):
        payload[f"category_level{level}"] = name
        payload[f"category_id_level{level}"] = category_ids[level]
    return category_ids[-1], payload


def _category_payload_needs_update(
    existing: Dict[str, Any],
    expected: Dict[str, Any],
) -> bool:
    """Проверяет, отличается ли payload существующего узла от ожидаемого."""
    # Даже при parent_id=None ключ должен присутствовать: старые узлы могли
    # быть созданы без него.
    if "parent_id" not in existing:
        return True

    for key, value in expected.items():
        if existing.get(key) != value:
            return True

    # Лишние уровни (например, после исправления пути) нужно удалить.
    for key in existing:
        if key.startswith("category_level") or key.startswith("category_id_level"):
            if key not in expected:
                return True

    return False


def _merge_category_payload(
    existing: Optional[Dict[str, Any]],
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    """Сохраняет прочие поля payload и заменяет категориальные уровни."""
    if not existing:
        return dict(expected)

    merged = {
        key: value
        for key, value in existing.items()
        if not (key.startswith("category_level") or key.startswith("category_id_level"))
    }
    merged.update(expected)
    return merged


def _build_category_nodes(
    paths: List[List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Достраивает все родительские узлы для набора путей категорий."""
    nodes: Dict[str, Dict[str, Any]] = {}
    for parts in paths:
        for depth in range(1, len(parts) + 1):
            prefix = parts[:depth]
            full_path = " / ".join(prefix)
            if full_path not in nodes:
                category_id, payload = _build_expected_category_payload(prefix)
                nodes[full_path] = {"id": category_id, "payload": payload}
    return nodes


async def _collect_category_paths(
    client,
    collection_name: str,
    include_old_versions: bool,
    has_is_latest: bool,
) -> Tuple[List[List[str]], str, List[Dict[str, Any]], int]:
    """Собирает уникальные пути категорий из коллекции-источника.

    Всегда используется scroll: уровни берутся из ``category_levelN``, потому
    что название уровня может само содержать разделитель " / ". Facet по
    ``category_path`` такой информации не даёт и приводит к ложным узлам.

    Заодно находит чанки, у которых ``category_id_levelN`` не совпадают с
    детерминированными ID узлов для их пути. Так бывает после переноса
    коллекции с другой машины: пути те же, а ID категорий в чанках — от
    прежней коллекции ``categories``.

    Returns:
        (пути, режим сканирования, группы исправлений, прочитано точек).
        Группа исправления — словарь с ``payload`` (новые ID) и ``points``.
    """
    scroll_filter = None
    if not include_old_versions and has_is_latest:
        scroll_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="is_latest",
                    match=qdrant_models.MatchValue(value=True),
                )
            ]
        )

    level_fields = [
        f"category_level{level}"
        for level in range(_CATEGORY_MAX_RECONSTRUCT_LEVELS)
    ]
    id_fields = [
        f"category_id_level{level}"
        for level in range(_CATEGORY_MAX_RECONSTRUCT_LEVELS)
    ]
    unique: Dict[Tuple[str, ...], List[str]] = {}
    id_fix_groups: Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]] = {}
    points_scanned = 0
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=_CATEGORY_SCAN_BATCH,
            offset=offset,
            with_payload=["category_path", "category_id", "source_id"]
            + level_fields
            + id_fields,
            with_vectors=False,
        )
        points_scanned += len(points)
        for point in points:
            payload = point.payload or {}
            parts = _payload_category_parts(payload)
            if not parts:
                continue
            unique.setdefault(tuple(parts), parts)

            if not _has_explicit_category_levels(payload):
                continue
            expected_ids = _category_ids_for_parts(parts)
            current_ids = [
                payload.get(f"category_id_level{level}")
                for level in range(len(parts))
            ]
            if current_ids == expected_ids:
                continue

            update = {
                f"category_id_level{level}": category_id
                for level, category_id in enumerate(expected_ids)
            }
            # Если поле category_id в чанках есть, оно тоже должно указывать
            # на листовой узел.
            if "category_id" in payload:
                update["category_id"] = expected_ids[-1]

            group_key = tuple(sorted(update.items()))
            group = id_fix_groups.get(group_key)
            if group is None:
                group = {"payload": update, "points": [], "sample_paths": []}
                id_fix_groups[group_key] = group
            group["points"].append(point.id)
            if len(group["sample_paths"]) < 2:
                group["sample_paths"].append(" / ".join(parts))

        if offset is None:
            break

    return list(unique.values()), "scroll", list(id_fix_groups.values()), points_scanned


async def _apply_category_id_fixes(
    client,
    collection_name: str,
    id_fix_groups: List[Dict[str, Any]],
) -> None:
    """Обновляет ``category_id_levelN`` в чанках коллекции-источника."""
    operations = []
    for group in id_fix_groups:
        point_ids = group["points"]
        for start in range(0, len(point_ids), _CATEGORY_SET_PAYLOAD_BATCH):
            operations.append(
                qdrant_models.SetPayloadOperation(
                    set_payload=qdrant_models.SetPayload(
                        payload=group["payload"],
                        points=point_ids[start:start + _CATEGORY_SET_PAYLOAD_BATCH],
                    )
                )
            )

    for start in range(0, len(operations), _CATEGORY_BATCH_UPDATE_BATCH):
        await client.batch_update_points(
            collection_name=collection_name,
            update_operations=operations[start:start + _CATEGORY_BATCH_UPDATE_BATCH],
            wait=True,
        )



async def _retrieve_existing_category_payloads(
    client,
    ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Батчево читает payload существующих узлов коллекции categories."""
    existing: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(ids), _CATEGORY_RETRIEVE_BATCH):
        batch = ids[start:start + _CATEGORY_RETRIEVE_BATCH]
        points = await client.retrieve(
            collection_name=settings.category_collection,
            ids=batch,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            existing[str(point.id)] = point.payload or {}
    return existing


async def sync_categories_from_collection(
    collection_name: str,
    include_old_versions: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Актуализирует коллекцию categories по категориям указанной коллекции.

    Создаёт отсутствующие узлы (включая всех родителей), исправляет узлы с
    некорректным или устаревшим payload. Дополнительно приводит
    ``category_id_levelN`` в чанках коллекции-источника к детерминированным
    ID узлов для их пути — это чинит рассинхронизацию ID, возникшую после
    переноса коллекции с другой машины (когда пути те же, а ID категорий
    взяты из прежней коллекции ``categories``).

    Операция идемпотентна: повторный запуск на неизменной коллекции ничего
    не пишет и не считает эмбеддинги.

    Args:
        collection_name: коллекция-источник (не должна быть category_collection).
        include_old_versions: учитывать точки с ``is_latest=false``.
        dry_run: только посчитать изменения, не писать в Qdrant.

    Returns:
        Словарь со статистикой операции.
    """
    started = time.monotonic()
    client = get_client()

    if not collection_name:
        raise ValueError("Collection name is required")
    if collection_name == settings.category_collection:
        raise ValueError(
            f"Collection '{collection_name}' is the target category collection"
        )

    try:
        collection_info = await client.get_collection(collection_name)
    except UnexpectedResponse as exc:
        if "Not found: Collection" in str(exc):
            raise ValueError(f"Collection '{collection_name}' not found") from exc
        raise

    payload_schema = getattr(collection_info, "payload_schema", {}) or {}
    has_is_latest = "is_latest" in payload_schema

    source_paths, scan_mode, id_fix_groups, points_scanned = await _collect_category_paths(
        client=client,
        collection_name=collection_name,
        include_old_versions=include_old_versions,
        has_is_latest=has_is_latest,
    )
    nodes = _build_category_nodes(source_paths)

    source_path_set = {" / ".join(parts) for parts in source_paths}
    parent_nodes = sum(1 for path in nodes if path not in source_path_set)

    categories_exists = True
    try:
        await client.get_collection(settings.category_collection)
    except UnexpectedResponse as exc:
        if "Not found: Collection" in str(exc):
            categories_exists = False
        else:
            raise

    if not categories_exists and not dry_run and nodes:
        # Переиспользуем штатное создание коллекции с payload-индексами.
        await QdrantBatchWriter._ensure_collection(settings.category_collection)
        categories_exists = True

    existing: Dict[str, Dict[str, Any]] = {}
    if categories_exists and nodes:
        existing = await _retrieve_existing_category_payloads(
            client,
            [node["id"] for node in nodes.values()],
        )

    missing: List[Tuple[str, str, Dict[str, Any]]] = []
    outdated: List[Tuple[str, str, Dict[str, Any]]] = []
    unchanged = 0

    for full_path, node in nodes.items():
        category_id = node["id"]
        expected_payload = node["payload"]
        current_payload = existing.get(category_id)

        if current_payload is None:
            missing.append((full_path, category_id, expected_payload))
        elif _category_payload_needs_update(current_payload, expected_payload):
            outdated.append((full_path, category_id, expected_payload))
        else:
            unchanged += 1

    created = 0
    updated = 0
    embedded = 0

    if not dry_run:
        to_write = missing + outdated
        for start in range(0, len(to_write), _CATEGORY_UPSERT_BATCH):
            chunk = to_write[start:start + _CATEGORY_UPSERT_BATCH]
            paths_to_encode = [full_path for full_path, _, _ in chunk]
            vectors = await encode_text(paths_to_encode, task_type="document")  # type: ignore

            if len(vectors) != len(chunk):
                raise RuntimeError(
                    "encode_text returned "
                    f"{len(vectors)} vectors for {len(chunk)} category paths"
                )

            points = []
            for (full_path, category_id, expected_payload), vector in zip(chunk, vectors):
                current_payload = existing.get(category_id)
                payload = _merge_category_payload(current_payload, expected_payload)
                points.append(
                    qdrant_models.PointStruct(
                        id=category_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            await client.upsert(
                collection_name=settings.category_collection,
                points=points,
                wait=True,
            )
            embedded += len(points)

        created = len(missing)
        updated = len(outdated)

    points_with_wrong_ids = sum(len(group["points"]) for group in id_fix_groups)
    points_updated = 0
    if not dry_run and id_fix_groups:
        await _apply_category_id_fixes(client, collection_name, id_fix_groups)
        points_updated = points_with_wrong_ids

    sample_wrong_ids = sorted(
        (
            f"{group['sample_paths'][0]} ({len(group['points'])} точек)"
            for group in id_fix_groups
            if group["sample_paths"]
        ),
    )[:_CATEGORY_PREVIEW_LIMIT]

    return {
        "collection_name": collection_name,
        "dry_run": dry_run,
        "include_old_versions": include_old_versions,
        "scan_mode": scan_mode,
        "points_scanned": points_scanned,
        "source_category_paths": len(source_paths),
        "category_nodes": len(nodes),
        "parent_nodes": parent_nodes,
        "missing": len(missing),
        "outdated": len(outdated),
        "unchanged": unchanged,
        "created": created,
        "updated": updated,
        "embedded": embedded,
        "points_with_wrong_category_ids": points_with_wrong_ids,
        "points_updated": points_updated,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "sample_missing": [
            path for path, _, _ in sorted(missing, key=lambda item: item[0])
        ][:_CATEGORY_PREVIEW_LIMIT],
        "sample_outdated": [
            path for path, _, _ in sorted(outdated, key=lambda item: item[0])
        ][:_CATEGORY_PREVIEW_LIMIT],
        "sample_wrong_category_ids": sample_wrong_ids,
    }


async def search_categories(
    query_text: str,
    limit: int = 10,
    min_score: Optional[float] = settings.search_min_score,
    exact_phrases: Optional[List[str]] = None,
    phrase_mode: str = "any",
) -> List[CategoryResponse]:
    """
    Выполняет семантический поиск по категориям на основе текстового запроса.

    Аргументы:
        query_text (str): Текст запроса для поиска
        limit (int): Максимальное количество возвращаемых результатов (по умолчанию 10)
        min_score (float | None): Минимальный score категории (0..1); меньшие отбрасываются
        exact_phrases (list[str] | None): Фразы для точного поиска по category_path
        phrase_mode (str): "any" или "all" - режим exact-поиска

    Возвращает:
        List[CategoryResponse]: Список найденных категорий с их метаданными и оценкой релевантности
    """
    if not query_text.strip():
        return []

    if min_score is None:
        min_score = settings.search_min_score

    client = get_client()
    normalized_query = basic_normalize(query_text)
    query_vector = await encode_text(normalized_query, task_type="query")  # type: ignore

    phrases = normalize_phrases(exact_phrases)
    # При exact-поиске берём запас кандидатов: точное совпадение может иметь
    # невысокое косинусное сходство и не попасть в обычный top-limit.
    candidate_limit = min(max(limit * 5, limit), 200) if phrases else limit

    try:
        response = await client.query_points(
            collection_name=settings.category_collection,
            query=query_vector,
            limit=candidate_limit,
            # Порог применяем в Python: точные совпадения он не должен отсекать
            score_threshold=None if phrases else min_score,
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = point.payload
            category_path = payload.get("category_path", "")
            categories = category_path.split(" / ") if category_path else []

            # Преобразуем point.id в строку, если это list[float] (например, qdrant)
            point_id = point.id
            if isinstance(point_id, list):
                point_id = ",".join(str(x) for x in point_id)
            else:
                point_id = str(point_id)

            found = matched_phrases(category_path, phrases)
            if phrases and phrase_mode == "all":
                exact_match = len(found) == len(phrases)
            else:
                exact_match = bool(found)

            result = CategoryResponse(
                id=point_id,
                score=point.score,
                category_name=payload.get("category_name", ""),
                category_path=category_path,
                categories=categories,
                category_level=payload.get("category_level", 0),
                category_id=payload.get("category_id"),
                levels={k: v for k, v in payload.items() if k.startswith("category_level") and isinstance(v, str)},
                id_levels={k: v for k, v in payload.items() if k.startswith("category_id_level")},
                exact_match=True if exact_match else None,
                matched_phrases=found or None,
            )
            results.append(result)

        if phrases:
            # Exact-совпадения первыми, затем по числу фраз и score
            results.sort(
                key=lambda c: (
                    0 if c.exact_match else 1,
                    -len(c.matched_phrases or []),
                    -c.score,
                )
            )
            results = [
                cat for cat in results
                if cat.exact_match or min_score is None or cat.score >= min_score
            ]

        return results[:limit]

    except UnexpectedResponse as e:
        if "Not found: Collection" in str(e):
            return []
        raise


def _get_field_value(cat, field: str) -> Any:
    """
    Получает значение поля из CategoryResponse.
    Поддерживает как прямые атрибуты, так и поля из levels/id_levels.
    
    Args:
        cat: CategoryResponse
        field: Имя поля (path, score, category_name, levels.category_level0, и т.д.)
    
    Returns:
        Значение поля или None
    """
    # Прямые атрибуты
    if field == "path":
        return cat.category_path
    elif field == "score":
        return cat.score
    elif field == "category_name":
        return cat.category_name
    elif field == "categories":
        return cat.categories
    elif field == "category_level":
        return cat.category_level
    elif field == "category_id":
        return cat.category_id
    elif field == "id":
        return cat.id
    elif field == "exact_match":
        return cat.exact_match
    elif field == "matched_phrases":
        return cat.matched_phrases

    # Поля из levels
    if field.startswith("levels."):
        level_key = field.replace("levels.", "")
        return cat.levels.get(level_key)
    
    # Поля из id_levels
    if field.startswith("id_levels."):
        level_key = field.replace("id_levels.", "")
        return cat.id_levels.get(level_key)
    
    return None

def _default_category_item(cat: CategoryResponse, with_exact: bool) -> Dict[str, Any]:
    """Компактный элемент ответа: path и score, плюс пометка exact-совпадения."""
    item: Dict[str, Any] = {"path": cat.category_path, "score": cat.score}
    if with_exact and cat.exact_match:
        item["exact_match"] = True
        item["matched_phrases"] = cat.matched_phrases or []
    return item


async def search_categories_formatted(
    query_text: str,
    limit: int = 10,
    grouped: bool = False,
    fields: Optional[List[str]] = None,
    min_score: Optional[float] = settings.search_min_score,
    exact_phrases: Optional[List[str]] = None,
    phrase_mode: str = "any",
) -> Any:
    """
    Универсальная функция поиска категорий с разными форматами вывода.

    Аргументы:
        query_text: Текст запроса
        limit: Максимальное количество результатов
        grouped: Группировать ли по коллекциям
        fields: Список полей для возврата (если None, возвращаются только path и score)
        min_score: Минимальный score категории (0..1); меньшие отбрасываются
        exact_phrases: Ключевые фразы для точного поиска по путям категорий
        phrase_mode: any - достаточно любой фразы, all - все фразы

    Возвращает:
        Dict[str, Any]: Результаты в требуемом формате
    """
    if grouped:
        # Получаем группированные результаты
        results_dict = await search_categories_by_collections(
            query_text, limit, min_score=min_score,
            exact_phrases=exact_phrases, phrase_mode=phrase_mode,
        )

        if fields is None:
            # Возвращаем только пути с score (по умолчанию); при exact-поиске
            # помечаем совпадения, чтобы клиент мог их отличить.
            formatted_dict = {}
            for collection_name, categories in results_dict.items():
                formatted_dict[collection_name] = [
                    _default_category_item(cat, bool(exact_phrases))
                    for cat in categories
                ]
            return formatted_dict
        else:
            # Формируем ответ с указанными полями
            formatted_dict = {}
            for collection_name, categories in results_dict.items():
                formatted_dict[collection_name] = [
                    {field: _get_field_value(cat, field) for field in fields}
                    for cat in categories
                ]
            return formatted_dict
    else:
        # Получаем не группированные результаты
        results = await search_categories(
            query_text, limit, min_score=min_score,
            exact_phrases=exact_phrases, phrase_mode=phrase_mode,
        )

        if fields is None:
            # Возвращаем только пути с score (по умолчанию); при exact-поиске
            # помечаем совпадения, чтобы клиент мог их отличить.
            return [
                _default_category_item(cat, bool(exact_phrases))
                for cat in results
            ]
        else:
            # Формируем ответ с указанными полями
            return [
                {field: _get_field_value(cat, field) for field in fields}
                for cat in results
            ]


async def search_categories_by_collections(
    query_text: str,
    limit: int = 10,
    min_score: Optional[float] = settings.search_min_score,
    exact_phrases: Optional[List[str]] = None,
    phrase_mode: str = "any",
) -> Dict[str, List[CategoryResponse]]:
    """
    Выполняет поиск категорий и группирует результаты по коллекциям.

    Аргументы:
        query_text (str): Текст запроса для поиска
        limit (int): Максимальное количество релевантных категорий (по умолчанию 10)
        min_score (float | None): Минимальный score категории (0..1); меньшие отбрасываются

    Возвращает:
        Dict[str, List[CategoryResponse]]: Словарь, где ключи - имена коллекций,
        а значения - списки категорий, отсортированные по score (убывание)
    """
    if not query_text.strip():
        return {}

    client = get_client()

    # Получаем релевантные категории по запросу
    relevant_categories = await search_categories(
        query_text, limit, min_score=min_score,
        exact_phrases=exact_phrases, phrase_mode=phrase_mode,
    )
    
    if not relevant_categories:
        return {}
    
    # Создаем словарь для быстрого поиска категории по path
    relevant_categories_by_path = {
        cat.category_path: cat for cat in relevant_categories
    }
    relevant_paths = set(relevant_categories_by_path.keys())
    
    # Получаем список всех коллекций
    try:
        collections_response = await client.get_collections()
        collection_names = [
            c.name for c in collections_response.collections
            if c.name != settings.category_collection
        ]
    except UnexpectedResponse as e:
        if "Not found: Collection" in str(e):
            return {}
        raise
    except Exception as e:
        # В случае других ошибок возвращаем пустой результат
        logger.warning(f"Failed to get collections: {e}")
        return {}
    
    if not collection_names:
        return {}
    
    # Для каждой коллекции находим категории через query groups
    collection_categories: Dict[str, Dict[str, float]] = {}
    
    for collection_name in collection_names:
        try:
            collection_categories[collection_name] = await _get_categories_by_facet(
                client, collection_name, relevant_paths, relevant_categories_by_path
            )
        except UnexpectedResponse as e:
            # Пропускаем коллекции, которые не существуют
            if "Not found: Collection" in str(e):
                continue
            # Для других UnexpectedResponse также продолжаем (на всякий случай)
            continue
        except Exception as e:
            # Пропускаем коллекции с другими ошибками
            logger.warning(f"Failed to process collection {collection_name}: {e}")
            continue
    
    # Определяем, в каких коллекциях присутствует каждая категория
    collection_category_paths: Dict[str, set] = {}
    category_collection_map: Dict[str, set] = {}
    
    for collection_name, categories_dict in collection_categories.items():
        if not categories_dict:
            continue
        
        for category_path in categories_dict.keys():
            matched_cat = None
            for cat in relevant_categories:
                if cat.category_path == category_path:
                    matched_cat = cat
                    break
                elif category_path.startswith(cat.category_path + " / "):
                    matched_cat = cat
                    break
            
            if matched_cat:
                if collection_name not in collection_category_paths:
                    collection_category_paths[collection_name] = set()
                collection_category_paths[collection_name].add(matched_cat.category_path)
                
                if matched_cat.category_path not in category_collection_map:
                    category_collection_map[matched_cat.category_path] = set()
                category_collection_map[matched_cat.category_path].add(collection_name)
                
                # Добавляем дочерние категории
                for cat in relevant_categories:
                    if cat.category_path != category_path:
                        if cat.category_path.startswith(category_path + " / "):
                            if cat.category_path not in collection_category_paths[collection_name]:
                                collection_category_paths[collection_name].add(cat.category_path)
                                if cat.category_path not in category_collection_map:
                                    category_collection_map[cat.category_path] = set()
                                category_collection_map[cat.category_path].add(collection_name)
    
    # Собираем все уникальные найденные категории
    found_category_paths = set(category_collection_map.keys())
    
    # Добавляем родителей найденных категорий
    for cat_path in list(found_category_paths):
        parts = cat_path.split(" / ")
        for i in range(1, len(parts)):
            parent_path = " / ".join(parts[:i])
            found_category_paths.add(parent_path)
    
    # Находим категории, которые не нашлись ни в одной коллекции
    not_found_categories = []
    for cat in relevant_categories:
        if cat.category_path not in found_category_paths:
            not_found_categories.append(("not_found_in_documents", cat))
    
    # Группируем категории по коллекциям
    result = {}
    for collection_name, category_paths in collection_category_paths.items():
        categories = []
        for cat_path in category_paths:
            if cat_path in relevant_categories_by_path:
                categories.append(relevant_categories_by_path[cat_path])
        
        categories.sort(
            key=lambda x: (
                0 if x.exact_match else 1,
                -len(x.matched_phrases or []),
                -x.score,
            )
        )
        result[collection_name] = categories
    
    # Добавляем не найденные категории
    if not_found_categories:
        not_found_categories.sort(key=lambda x: x[1].score, reverse=True)
        result["not_found_in_documents"] = [cat for _, cat in not_found_categories]
    
    return result


async def _get_categories_by_facet(
    client, collection_name: str, relevant_paths: set, relevant_categories_by_path: dict
) -> Dict[str, float]:
    """
    Метод для получения категорий через scroll с фильтром по category_id_level*.
    Находит категории, которые есть в указанных релевантных путях.
    
    Возвращает:
        Dict[str, float]: Словарь где ключи - category_path (строки в формате "cat1 / cat2"),
        значения - count (количество упоминаний)
    """
    if not relevant_paths:
        return {}
    
    relevant_ids_set = {cat.category_id for cat in relevant_categories_by_path.values() if cat.category_id}
    
    if not relevant_ids_set:
        return {}
    
    # Группируем category_id по их category_level
    ids_by_level: Dict[int, List[str]] = {}
    
    for cat in relevant_categories_by_path.values():
        if not cat.category_id:
            continue
        
        for key, value in cat.id_levels.items():
            if value == cat.category_id:
                level = int(key.replace("category_id_level", ""))
                if level not in ids_by_level:
                    ids_by_level[level] = []
                ids_by_level[level].append(cat.category_id)
                break
        else:
            if 0 not in ids_by_level:
                ids_by_level[0] = []
            ids_by_level[0].append(cat.category_id)
    
    # Строим filter по category_id_level*
    should_conditions = []
    
    for level, ids in ids_by_level.items():
        if ids:
            should_conditions.append(
                qdrant_models.FieldCondition(
                    key=f"category_id_level{level}",
                    match=qdrant_models.MatchAny(any=ids)
                )
            )
    
    if not should_conditions:
        return {}
    
    query_filter = qdrant_models.Filter(
        should=should_conditions
    )
    
    points = await scroll_all_pages(
        client=client,
        collection_name=collection_name,
        scroll_filter=query_filter,
        limit=10000,
        with_payload=True,
    )
    
    category_ids_found: set = set()
    
    for point in points:
        if not point.payload:
            continue
        
        cat_id = point.payload.get("category_id")
        if cat_id and cat_id in relevant_ids_set:
            category_ids_found.add(cat_id)
        
        for key, value in point.payload.items():
            if key.startswith("category_id_level") and value and isinstance(value, str):
                if value in relevant_ids_set:
                    category_ids_found.add(value)
    
    # Собираем финальные результаты
    category_counts: Dict[str, int] = {}
    for cat_id in category_ids_found:
        for cat in relevant_categories_by_path.values():
            if cat.category_id == cat_id:
                category_counts[cat.category_path] = category_counts.get(cat.category_path, 0) + 1
                break
    
    return {path: float(count) for path, count in category_counts.items()}


@router.post("/search",
    summary="Поиск по категориям",
    description=load_openapi_md("categories_search.md"))
async def search_categories_endpoint(
    body: CategorySearchRequest,
    grouped: bool = Query(False, description="Если True, возвращает категории, сгруппированные по коллекциям")
):
    """
    Выполняет поиск категорий по текстовому запросу.
    
    Аргументы:
        body: Запрос с query_text, limit и fields
        grouped: Если True, возвращает категории, сгруппированные по коллекциям
    
    По умолчанию возвращает только `path` и `score`. 
    Для получения дополнительных полей укажите их в поле `fields` запроса.
    """
    results = await search_categories_formatted(
        query_text=body.query_text,
        limit=body.limit,
        grouped=grouped,
        fields=body.fields,
        min_score=body.min_score,
        exact_phrases=body.exact_phrases,
        phrase_mode=body.phrase_mode,
    )
    return {'results': results}


@router.get("",
    summary="Получение всех категорий",
    description=load_openapi_md("categories_list.md"))
async def get_all_categories(
    limit: int = Query(100, description="Максимальное количество результатов (1-1000)", ge=1, le=1000, examples=[50]),
    parent_id: Optional[str] = Query(None, description="Фильтр по родительской категории", examples=["uuid-категории-родителя"]),
):
    """
    Получает все категории из коллекции.
    """
    client = get_client()
    
    try:
        # Строим фильтр если указан parent_id
        scroll_filter = None
        if parent_id is not None:
            scroll_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="parent_id",
                        match=qdrant_models.MatchValue(value=parent_id)
                    )
                ]
            )
        
        points, _ = await client.scroll(
            collection_name=settings.category_collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )
        
        results = []
        for point in points:
            payload = point.payload
            category_path = payload.get("category_path", "")
            categories = category_path.split(" / ") if category_path else []
            
            result = CategoryResponse(
                id=point.id,
                score=1.0,  # scroll не возвращает score, ставим 1.0
                category_name=payload.get("category_name", ""),
                category_path=category_path,
                categories=categories,
                category_level=payload.get("category_level", 0),
                category_id=payload.get("category_id"),
                levels={k: v for k, v in payload.items() if k.startswith("category_level") and isinstance(v, str)},
                id_levels={k: v for k, v in payload.items() if k.startswith("category_id_level")}
            )
            results.append(result)
        
        return CategoriesSearchResponse(results=results)
        
    except UnexpectedResponse as e:
        if "Not found: Collection" in str(e):
            return CategoriesSearchResponse(results=[])
        raise
