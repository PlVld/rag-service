from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Form
from app.auth import verify_api_key
from pydantic import BaseModel, Field
from qdrant_client.http import models as qdrant_models
from app.api.health import get_client
from app.api.categories import sync_categories_from_collection
from app.core.config import settings
from app.openapi_md.loader import load_openapi_md
import logging

router = APIRouter(prefix="/v1/admin",
                   tags=["Admin"],
                   dependencies=[Depends(verify_api_key)],)
logger = logging.getLogger(__name__)


class CollectionCategories(BaseModel):
    name: str
    categories: List[str]


class CollectionsWithCategoriesResponse(BaseModel):
    results: List[CollectionCategories]


class CategoryChunkCount(BaseModel):
    category_path: str
    chunk_count: int


class CategoryHierarchyRequest(BaseModel):
    collection_name: Optional[str] = Field(None, description="Имя коллекции (опционально)")
    depth: int = Field(0, ge=0, le=10, description="Количество уровней для получения (0 - только список коллекций без категорий, 1-10 - иерархия категорий). Максимальное значение: 10."),
    categories: Optional[List[str]] = Field(None, description="Массив полных путей категорий (опционально). Разделитель уровней - \" / \", также принимается \"/\" без пробелов. Если указаны, возвращаются только коллекции с этими категориями.")

    limit: Optional[int] = Field(
        5000,
        ge=1,
        description=(
            "Максимальное суммарное количество категорий в ответе. "
            "Коллекции обходятся по порядку, пока не исчерпан лимит. "
            "null - без ограничения."
        ),
    )


class CollectionCategoryHierarchy(BaseModel):
    name: str
    categories: List[CategoryChunkCount]
    total_categories: int = Field(0, description="Категорий в коллекции до применения limit")
    truncated: bool = Field(False, description="Список категорий коллекции обрезан лимитом")


class CategoryHierarchyResponse(BaseModel):
    results: List[CollectionCategoryHierarchy]
    limit: Optional[int] = Field(None, description="Лимит на весь ответ; null - без ограничения")
    truncated: bool = Field(False, description="Ответ обрезан лимитом")
    total_categories: int = Field(0, description="Всего категорий до применения limit")
    shown_categories: int = Field(0, description="Категорий в ответе после применения limit")


class HnswConfigRequest(BaseModel):
    collection_name: str
    m: int = 16


class PayloadIndexRequest(BaseModel):
    collection_name: str
    field_name: str
    field_type: qdrant_models.PayloadSchemaType = qdrant_models.PayloadSchemaType.KEYWORD


class CategorySyncRequest(BaseModel):
    collection_name: str = Field(
        ...,
        min_length=1,
        description="Имя коллекции-источника, из которой берутся paths категорий",
        examples=["1c_integration"],
    )
    include_old_versions: bool = Field(
        False,
        description="Учитывать точки с is_latest=false (по умолчанию только актуальные)",
    )
    dry_run: bool = Field(
        False,
        description="Только посчитать изменения, не создавать эмбеддинги и не писать в Qdrant",
    )


class CategorySyncResponse(BaseModel):
    collection_name: str
    dry_run: bool
    include_old_versions: bool
    scan_mode: str = Field(..., description="Способ чтения путей: scroll")
    points_scanned: int = Field(..., description="Прочитано точек коллекции-источника")
    source_category_paths: int = Field(..., description="Уникальных путей в коллекции-источнике")
    category_nodes: int = Field(..., description="Узлов с учётом всех родителей")
    parent_nodes: int = Field(..., description="Узлов, которых нет среди путей источника")
    missing: int = Field(..., description="Узлов отсутствует в categories")
    outdated: int = Field(..., description="Узлов с некорректным payload")
    unchanged: int = Field(..., description="Узлов без изменений")
    created: int = Field(..., description="Создано узлов (0 при dry_run)")
    updated: int = Field(..., description="Обновлено узлов (0 при dry_run)")
    embedded: int = Field(..., description="Посчитано эмбеддингов (0 при dry_run)")
    points_with_wrong_category_ids: int = Field(
        ...,
        description="Точек, у которых category_id_levelN не совпадает с ID узлов для их пути",
    )
    points_updated: int = Field(
        ...,
        description="Точек с исправленными category_id_levelN (0 при dry_run)",
    )
    duration_ms: int
    sample_missing: List[str] = Field(default_factory=list)
    sample_outdated: List[str] = Field(default_factory=list)
    sample_wrong_category_ids: List[str] = Field(default_factory=list)


@router.post("/hnsw", summary="Включить/выключить HNSW-индекс", description=load_openapi_md("admin_hnsw.md"))
async def set_hnsw(
    collection_name: str = Form(..., description="Имя коллекции"),
    enabled: bool = Form(..., description="Включить (true) или выключить (false) HNSW"),
    m: int = Form(16, ge=0, description="Параметр HNSW m (используется только при enabled=true)"),
):
    client = get_client()
    try:
        effective_m = m if enabled else 0
        await client.update_collection(
            collection_name=collection_name,
            hnsw_config=qdrant_models.HnswConfigDiff(m=effective_m),
        )
        logger.info(f"HNSW updated for {collection_name}: enabled={enabled}, m={effective_m}")
        state = "enabled" if enabled else "disabled"
        return {"status": "ok", "message": f"HNSW {state} for {collection_name}", "enabled": enabled, "m": effective_m}
    except Exception as e:
        logger.error(f"Failed to update HNSW: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create_payload_index",
             summary="Создать индекс для поля payload",
             description=load_openapi_md("admin_create_payload_index.md"))
async def create_payload_index(request: PayloadIndexRequest):
    client = get_client()
    try:
        await client.create_payload_index(
            collection_name=request.collection_name,
            field_name=request.field_name,
            field_type=request.field_type
        )
        logger.info(f"Payload index created for {request.collection_name}.{request.field_name}")
        return {"status": "ok", "message": f"Index created for {request.field_name}"}
    except Exception as e:
        logger.error(f"Failed to create payload index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections", response_model=CollectionsWithCategoriesResponse)
async def get_collections_with_categories():
    client = get_client()
    try:
        collections_response = await client.get_collections()
        collection_names = [c.name for c in collections_response.collections
                            if c.name != settings.category_collection]

        result = []
        for collection_name in collection_names:
            try:
                facet_result = await client.facet(
                    collection_name=collection_name,
                    key="category_level0",
                    limit=10000,
                    facet_filter=qdrant_models.Filter(
                        must_not=[
                            qdrant_models.IsEmptyCondition(
                                is_empty=qdrant_models.PayloadField(key="category_level0")
                            )
                        ]
                    )
                )
                categories = {hit.value for hit in facet_result.hits if hit.value}
                result.append(CollectionCategories(name=collection_name, categories=sorted(categories)))
            except Exception as e:
                logger.error(f"Facet failed for {collection_name}: {e}")
                raise

        return CollectionsWithCategoriesResponse(results=result)
    except Exception as e:
        logger.error(f"Failed to get collections with categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_category_level(category_path: str) -> int:
    """Определяет уровень категории по количеству разделителей."""
    if not category_path:
        return 0
    return category_path.count(" / ")


def _normalize_category_path(category_path: str) -> str:
    """Приводит путь категории к каноническому виду с разделителем " / ".

    В БД разделитель всегда " / ", но пользователь может передать "/" без пробелов
    ("A/B/C"). Без нормализации full-text фильтр по такому пути не находит ни одного
    чанка, и эндпоинт молча возвращает пустой результат.
    """
    if " / " in category_path:
        return category_path.strip()
    return " / ".join(part.strip() for part in category_path.split("/") if part.strip())


def _build_parent_conditions(categories: Optional[List[str]], current_level: int) -> List[qdrant_models.FieldCondition]:
    """
    Строит список условий для родительских категорий (prefix match по category_path).
    Применяется только для уровней >= уровня родительской категории.
    Возвращает список FieldCondition, а не Filter.
    """
    if not categories:
        return []

    parent_conditions = []
    for cat_path in categories:
        cat_level = _get_category_level(cat_path)
        if current_level >= cat_level:
            parent_conditions.append(
                qdrant_models.FieldCondition(
                    key="category_path",
                    match=qdrant_models.MatchText(text=cat_path)
                )
            )

    return parent_conditions


async def _get_category_paths_by_ids(client, category_ids: List[str]) -> Dict[str, str]:
    """Получает полные пути категорий по их ID из коллекции categories."""
    if not category_ids:
        return {}
    try:
        points = await client.retrieve(
            collection_name=settings.category_collection,
            ids=category_ids,
            with_payload=["category_path"]
        )
        return {point.id: point.payload.get("category_path", "") for point in points}
    except Exception as e:
        logger.warning(f"Failed to retrieve category paths: {e}")
        return {}


def _apply_category_limit(
    collections_result: List[Dict[str, Any]],
    limit: Optional[int],
) -> Dict[str, Any]:
    """Ограничивает суммарное количество категорий в ответе.

    Лимит общий на весь ответ: коллекции обходятся по порядку, пока не
    исчерпан бюджет. В каждой коллекции проставляются ``total_categories``
    (сколько категорий было до обрезки) и ``truncated``.
    """
    total = sum(len(coll.get("categories", [])) for coll in collections_result)
    shown = 0
    truncated = False

    if limit is None:
        for coll in collections_result:
            cats = coll.get("categories", [])
            coll["total_categories"] = len(cats)
            coll["truncated"] = False
            shown += len(cats)
        return {
            "limit": None,
            "truncated": False,
            "total_categories": total,
            "shown_categories": shown,
        }

    remaining = max(limit, 0)
    for coll in collections_result:
        cats = coll.get("categories", [])
        coll["total_categories"] = len(cats)

        if remaining <= 0:
            coll["categories"] = []
            coll["truncated"] = bool(cats)
        elif len(cats) > remaining:
            coll["categories"] = cats[:remaining]
            coll["truncated"] = True
            shown += remaining
            remaining = 0
        else:
            coll["categories"] = cats
            coll["truncated"] = False
            shown += len(cats)
            remaining -= len(cats)

        truncated = truncated or coll["truncated"]

    return {
        "limit": limit,
        "truncated": truncated,
        "total_categories": total,
        "shown_categories": shown,
    }


async def _get_category_hierarchy_data(
    collection_name: Optional[str] = None,
    depth: int = 1,
    categories: Optional[List[str]] = None,
    limit: Optional[int] = 5000,
) -> Dict[str, Any]:
    """
    Внутренняя функция получения иерархии категорий (без FastAPI зависимостей).
    Используется как для API endpoint, так и для MCP tool.

    ``limit`` ограничивает суммарное число категорий в ответе (None - без
    ограничения). Коллекции обходятся по порядку, пока не исчерпан лимит.
    """
    from app.api.health import get_client
    from app.core.config import settings
    
    client = get_client()

    try:
        # Получаем список всех коллекций
        collections_response = await client.get_collections()
        collection_names = [c.name for c in collections_response.collections
                            if c.name != settings.category_collection]

        # Фильтруем по указанной коллекции
        if collection_name:
            if collection_name not in collection_names:
                raise ValueError(f"Collection '{collection_name}' not found")
            collection_names = [collection_name]

        # Если depth == 0, возвращаем только названия коллекций без категорий
        if depth == 0:
            return {
                "results": [
                    {"name": name, "categories": []}
                    for name in collection_names
                ],
                "limit": limit,
                "truncated": False,
                "total_categories": 0,
                "shown_categories": 0,
            }

        # Если указаны categories, нормализуем разделители и логируем для отладки
        if categories:
            categories = [_normalize_category_path(c) for c in categories]
            logger.info(f"Getting category hierarchy for {len(categories)} categories: {categories}")

        collections_result = []

        for coll_name in collection_names:
            # Определяем целевые уровни
            if categories:
                target_levels = set()
                for cat_path in categories:
                    base_level = _get_category_level(cat_path)
                    for i in range(depth):
                        target_levels.add(base_level + i)
                target_levels = sorted(target_levels)
            else:
                target_levels = list(range(depth))

            category_counts: Dict[str, int] = {}
            last_level: Optional[int] = None

            # Пропускаем, если нет целевых уровней
            if not target_levels:
                continue

            for level in target_levels:
                last_level = level
                field_name = f"category_id_level{level}"

                # Базовый фильтр: только is_latest = true
                filter_conditions = [
                    qdrant_models.FieldCondition(
                        key="is_latest",
                        match=qdrant_models.MatchValue(value=True)
                    )
                ]

                # Добавляем фильтр по родительским категориям, если они заданы
                parent_conditions = _build_parent_conditions(categories, level)
                if parent_conditions:
                    # Объединяем через must (логическое И)
                    filter_conditions.extend(parent_conditions)

                # Строим итоговый фильтр для facet
                facet_filter = qdrant_models.Filter(must=filter_conditions) if filter_conditions else None

                # Выполняем facet-запрос для текущего уровня
                try:
                    facet_result = await client.facet(
                        collection_name=coll_name,
                        key=field_name,
                        limit=10000,
                        facet_filter=facet_filter
                    )
                except Exception as e:
                    logger.error(f"Facet failed for {coll_name} level {last_level}: {e}")
                    continue

                # Собираем результаты: ID категории -> количество точек
                for hit in facet_result.hits:
                    if hit.value is not None:
                        cat_id = str(hit.value)  # ID категории как строка
                        category_counts[cat_id] = category_counts.get(cat_id, 0) + hit.count

            # Логируем, если нет категорий для этой коллекции и уровня
            if not category_counts:
                if categories:
                    logger.debug(f"No categories found for collection '{coll_name}' with filter {categories} at level {last_level}")
                else:
                    logger.debug(f"No categories found for collection '{coll_name}' at level {last_level}")

            # Если указаны categories и для этой коллекции не найдено категорий, пропускаем её
            if categories and not category_counts:
                logger.info(f"Skipping collection '{coll_name}' - no categories found matching {categories}")
                continue

            # Получаем пути категорий по ID
            category_ids = list(category_counts.keys())
            id_to_path = await _get_category_paths_by_ids(client, category_ids)

            # Формируем результат
            results = []
            for cat_id, count in category_counts.items():
                cat_path = id_to_path.get(cat_id, "")
                if cat_path:
                    results.append({"category_path": cat_path, "chunk_count": count})

            results.sort(key=lambda x: x["category_path"])

            collections_result.append({
                "name": coll_name,
                "categories": results
            })

        return {"results": collections_result, **_apply_category_limit(collections_result, limit)}

    except Exception as e:
        logger.exception(f"Failed to get category hierarchy: {e}")
        raise


@router.post("/categories/hierarchy",
             summary="Иерархия категорий с количеством чанков",
             description=load_openapi_md("admin_categories_hierarchy.md"),
             response_model=CategoryHierarchyResponse)
async def get_category_hierarchy(request: CategoryHierarchyRequest):
    """
    Получает иерархию категорий с количеством чанков, используя facet-запросы.
    Если collection_name не указан, возвращаются все коллекции (кроме categories).
    При depth=0 возвращается только список коллекций без категорий.
    """
    try:
        result = await _get_category_hierarchy_data(
            collection_name=request.collection_name,
            depth=request.depth,
            categories=request.categories,
            limit=request.limit,
        )

        # Преобразуем результат в response model
        collections_result = []
        for coll in result["results"]:
            categories_list = []
            for cat in coll["categories"]:
                categories_list.append(CategoryChunkCount(
                    category_path=cat["category_path"],
                    chunk_count=cat["chunk_count"]
                ))
            collections_result.append(CollectionCategoryHierarchy(
                name=coll["name"],
                categories=categories_list,
                total_categories=coll.get("total_categories", len(categories_list)),
                truncated=coll.get("truncated", False),
            ))

        return CategoryHierarchyResponse(
            results=collections_result,
            limit=result.get("limit"),
            truncated=result.get("truncated", False),
            total_categories=result.get("total_categories", 0),
            shown_categories=result.get("shown_categories", 0),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to get category hierarchy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/categories/sync",
             summary="Актуализировать категории из коллекции",
             description=load_openapi_md("admin_categories_sync.md"),
             response_model=CategorySyncResponse)
async def sync_categories(request: CategorySyncRequest):
    """Создаёт/исправляет узлы коллекции categories по путям коллекции-источника.

    Используется после переноса коллекции с другой машины. Операция
    идемпотентна и не удаляет существующие категории.
    """
    if request.collection_name == settings.category_collection:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Collection '{request.collection_name}' is the target "
                "category collection"
            ),
        )

    try:
        result = await sync_categories_from_collection(
            collection_name=request.collection_name,
            include_old_versions=request.include_old_versions,
            dry_run=request.dry_run,
        )
        return CategorySyncResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(
            "Failed to sync categories from %s: %s",
            request.collection_name,
            e,
        )
        raise HTTPException(status_code=500, detail=str(e))