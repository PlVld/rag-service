from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Form
from app.auth import verify_api_key
from pydantic import BaseModel, Field
from qdrant_client.http import models as qdrant_models
from app.api.health import get_client
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
    depth: int = Field(0, ge=0, le=10, description="Количество уровней для получения (0 - только список коллекций без категорий, 1-10 - иерархия категорий). Максимальное значение: 10.")
    categories: Optional[List[str]] = Field(None, description="Массив полных путей категорий (опционально). Если указаны, возвращаются только коллекции с этими категориями.")


class CollectionCategoryHierarchy(BaseModel):
    name: str
    categories: List[CategoryChunkCount]


class CategoryHierarchyResponse(BaseModel):
    results: List[CollectionCategoryHierarchy]


class HnswConfigRequest(BaseModel):
    collection_name: str
    m: int = 16


class PayloadIndexRequest(BaseModel):
    collection_name: str
    field_name: str
    field_type: qdrant_models.PayloadSchemaType = qdrant_models.PayloadSchemaType.KEYWORD


@router.post("/hnsw", summary="Включить/выключить HNSW-индекс", description=load_openapi_md("admin_hnsw.md"))
async def set_hnsw(
    collection_name: str = Form(..., description="Имя коллекции"),
    enabled: bool = Form(..., description="Включить (true) или выключить (false) HNSW"),
    m: int = Form(16, ge=0, description="Параметр HNSW m (используется только при enabled=true)"),
):
    client = get_client()
    try:
        effective_m = m if enabled else 0
        client.update_collection(
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
        client.create_payload_index(
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
        collections_response = client.get_collections()
        collection_names = [c.name for c in collections_response.collections
                            if c.name != settings.category_collection]

        result = []
        for collection_name in collection_names:
            try:
                facet_result = client.facet(
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
        points = client.retrieve(
            collection_name=settings.category_collection,
            ids=category_ids,
            with_payload=["category_path"]
        )
        return {point.id: point.payload.get("category_path", "") for point in points}
    except Exception as e:
        logger.warning(f"Failed to retrieve category paths: {e}")
        return {}


async def _get_category_hierarchy_data(
    collection_name: Optional[str] = None,
    depth: int = 1,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Внутренняя функция получения иерархии категорий (без FastAPI зависимостей).
    Используется как для API endpoint, так и для MCP tool.
    """
    from app.api.health import get_client
    from app.core.config import settings
    
    client = get_client()

    try:
        # Получаем список всех коллекций
        collections_response = client.get_collections()
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
                ]
            }

        # Если указаны categories, логируем их для отладки
        if categories:
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
                    facet_result = client.facet(
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

        return {"results": collections_result}

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
                categories=categories_list
            ))
        
        return CategoryHierarchyResponse(results=collections_result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to get category hierarchy: {e}")
        raise HTTPException(status_code=500, detail=str(e))