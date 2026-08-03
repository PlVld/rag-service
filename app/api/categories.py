import logging
from typing import List, Optional, Dict, Tuple, Any
from fastapi import APIRouter, Query, Depends
from app.auth import verify_api_key
from app.openapi_md.loader import load_openapi_md

from pydantic import BaseModel, Field

from app.api.health import get_client
from app.core.config import settings
from app.utils import basic_normalize, generate_uuid_from_parts
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
    grouped: bool = Field(False, description="Если True, возвращает категории, сгруппированные по коллекциям", examples=[False])
    fields: Optional[List[str]] = Field(
        default=None,
        description=(
            "Список полей для возврата. Если не указан, возвращаются только `path` и `score`. "
            "Доступные поля: `id`, `score`, `category_name`, `category_path`, `categories`, "
            "`category_level`, `category_id`, `levels`, `id_levels`."
        ),
        examples=[["path", "score"]]
    )


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
        existing_points = client.retrieve(
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


async def search_categories(query_text: str, limit: int = 10) -> List[CategoryResponse]:
    """
    Выполняет семантический поиск по категориям на основе текстового запроса.
    
    Аргументы:
        query_text (str): Текст запроса для поиска
        limit (int): Максимальное количество возвращаемых результатов (по умолчанию 10)
    
    Возвращает:
        List[CategoryResponse]: Список найденных категорий с их метаданными и оценкой релевантности
    """
    if not query_text.strip():
        return []
    
    client = get_client()
    normalized_query = basic_normalize(query_text)
    query_vector = await encode_text(normalized_query, task_type="query")  # type: ignore
    
    try:
        response = client.query_points(
            collection_name=settings.category_collection,
            query=query_vector,
            limit=limit,
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
            
            result = CategoryResponse(
                id=point_id,
                score=point.score,
                category_name=payload.get("category_name", ""),
                category_path=category_path,
                categories=categories,
                category_level=payload.get("category_level", 0),
                category_id=payload.get("category_id"),
                levels={k: v for k, v in payload.items() if k.startswith("category_level") and isinstance(v, str)},
                id_levels={k: v for k, v in payload.items() if k.startswith("category_id_level")}
            )
            results.append(result)
        
        return results
        
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
    
    # Поля из levels
    if field.startswith("levels."):
        level_key = field.replace("levels.", "")
        return cat.levels.get(level_key)
    
    # Поля из id_levels
    if field.startswith("id_levels."):
        level_key = field.replace("id_levels.", "")
        return cat.id_levels.get(level_key)
    
    return None

async def search_categories_formatted(
    query_text: str,
    limit: int = 10,
    grouped: bool = False,
    fields: Optional[List[str]] = None,
) -> Any:
    """
    Универсальная функция поиска категорий с разными форматами вывода.
    
    Аргументы:
        query_text: Текст запроса
        limit: Максимальное количество результатов
        grouped: Группировать ли по коллекциям
        fields: Список полей для возврата (если None, возвращаются только path и score)
    
    Возвращает:
        Dict[str, Any]: Результаты в требуемом формате
    """
    if grouped:
        # Получаем группированные результаты
        results_dict = await search_categories_by_collections(query_text, limit)
        
        if fields is None:
            # Возвращаем только пути с score (по умолчанию)
            formatted_dict = {}
            for collection_name, categories in results_dict.items():
                formatted_dict[collection_name] = [
                    {"path": cat.category_path, "score": cat.score}
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
        results = await search_categories(query_text, limit)
        
        if fields is None:
            # Возвращаем только пути с score (по умолчанию)
            return [
                {"path": cat.category_path, "score": cat.score}
                for cat in results
            ]
        else:
            # Формируем ответ с указанными полями
            return [
                {field: _get_field_value(cat, field) for field in fields}
                for cat in results
            ]


async def search_categories_by_collections(query_text: str, limit: int = 10) -> Dict[str, List[CategoryResponse]]:
    """
    Выполняет поиск категорий и группирует результаты по коллекциям.
    
    Аргументы:
        query_text (str): Текст запроса для поиска
        limit (int): Максимальное количество релевантных категорий (по умолчанию 10)
    
    Возвращает:
        Dict[str, List[CategoryResponse]]: Словарь, где ключи - имена коллекций,
        а значения - списки категорий, отсортированные по score (убывание)
    """
    if not query_text.strip():
        return {}
    
    client = get_client()
    
    # Получаем релевантные категории по запросу
    relevant_categories = await search_categories(query_text, limit)
    
    if not relevant_categories:
        return {}
    
    # Создаем словарь для быстрого поиска категории по path
    relevant_categories_by_path = {
        cat.category_path: cat for cat in relevant_categories
    }
    relevant_paths = set(relevant_categories_by_path.keys())
    
    # Получаем список всех коллекций
    try:
        collections_response = client.get_collections()
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
        
        categories.sort(key=lambda x: x.score, reverse=True)
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
    
    points, _ = client.scroll(
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
        fields=body.fields
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
        
        points, _ = client.scroll(
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
