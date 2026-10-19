# app/mcp_server.py
# Utility functions and data models for MCP tools.
# 
# The MCP tools (search_documents_tool, search_categories_tool) 
# are now registered directly in app/main.py via FastApiMCP.
# This module contains only helper functions and data models used by those tools.
#
# If you want to hide or disable tools, configure ALLOWED_MCP_TOOLS in the environment
# (comma-separated names). See app/core/config.py for details.

import logging
from typing import Any, List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator

from app.api.categories import search_categories
from app.api.health import get_client
from app.api.documents import build_qdrant_filter, _perform_search, get_all_chunks, _matches_category_path
from app.api.documents_grouped import get_grouped_text
from app.api.documents_search import _group_search_results, _return_individual_chunks, rerank_by_query_similarity
from app.core.config import settings
from app.core.embeddings import encode_text
from app.core.exact_search import (
    exact_candidate_limit,
    filter_by_min_score,
    merge_exact_points,
    normalize_phrases,
    perform_exact_search,
    sort_hybrid_points,
)
from app.utils import basic_normalize

logger = logging.getLogger(__name__)


async def _prepare_search_metadata(
    query_text: str,
    filter_criteria: Optional[Dict[str, Any]] = None,
    include_old_versions: bool = False,
) -> Dict[str, Any]:
    """Подготовка метаданных для поиска: нормализация запроса, категории, вектор и фильтр."""
    normalized_query = basic_normalize(query_text)
    
    if not normalized_query:
        return {
            "success": True,
            "data": {"results": []},
            "prepared": None
        }
    
    # Получаем релевантные категории для RRF
    categories = await search_categories(normalized_query, limit=10)
    cat_ids = [cat.id for cat in categories]
    
    # Получаем вектор запроса
    query_vector = await encode_text(normalized_query, task_type="query")
    
    # Строим фильтр
    filter_conditions = filter_criteria or {}
    if not include_old_versions:
        filter_conditions["is_latest"] = True
    
    # Извлекаем category_path для последующей фильтрации
    category_path_filter = filter_conditions.pop("category_path", None)
    
    base_filter = build_qdrant_filter(filter_conditions)
    
    return {
        "success": False,
        "data": {},
        "prepared": {
            "normalized_query": normalized_query,
            "categories": categories,
            "cat_ids": cat_ids,
            "query_vector": query_vector,
            "filter_conditions": filter_conditions,
            "category_path_filter": category_path_filter,
            "base_filter": base_filter
        }
    }


# Определяем схемы для входных параметров (для автоматической валидации)
class SearchDocumentsInput(BaseModel):
    query_text: str = Field(
        ...,
        description="Текст запроса для семантического поиска",
        examples=["Как настроить векторный поиск?"]
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Имя коллекции в Qdrant. Если не указано, поиск выполняется по всем коллекциям, кроме служебной коллекции категорий.",
        examples=["documents"]
    )
    limit: int = Field(
        default=10,
        description="Макс. результатов (1-100)",
        ge=1,
        le=100,
        examples=[10]
    )
    min_score: Optional[float] = Field(
        default=settings.search_min_score,
        description=(
            "Минимальный score результата (0..1). Результаты с меньшим "
            "косинусным сходством отбрасываются. По умолчанию 0.55; "
            "null - тоже значение по умолчанию, 0 - без ограничения."
        ),
        ge=0.0,
        le=1.0,
        examples=[0.55, None]
    )
    exact_phrases: Optional[List[str]] = Field(
        default=None,
        max_length=10,
        description=(
            "Ключевые слова или фразы для точного (лексического) поиска, "
            "например имена методов и команды. Найденные совпадения добавляются "
            "к семантическим результатам и идут первыми. null или [] - только "
            "семантический поиск."
        ),
        examples=[["ЕСТЬNULL", "ISNULL / COALESCE"], None]
    )
    phrase_mode: Literal["any", "all"] = Field(
        default="any",
        description=(
            "Режим exact-поиска: any - достаточно любой фразы, all - должны "
            "встретиться все фразы."
        ),
        examples=["any", "all"]
    )
    filter_criteria: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="filter",
        description=(
            "Фильтр по payload в формате Qdrant.\\n"
            "Представляет собой объект с условиями точного совпадения (MatchValue).\\n"
            "Пример: `{\"category_path\": \"Синтаксис 1С / Таблицы значений\", \"is_latest\": true}`"
        )
    )
    
    # Свойство для доступа к filter по старому имени (для совместимости с тестами)
    @property
    def filter(self) -> Optional[Dict[str, Any]]:
        """Доступ к filter_criteria по старому имени для совместимости."""
        return self.filter_criteria
    
    include_old_versions: bool = Field(
        default=False,
        description="Включать старые версии (по умолчанию только актуальные)"
    )
    max_text_length: int = Field(
        default=0,
        description="Максимальная длина текста в результате (0 = без ограничений). Если задано положительное число, текст будет обрезан до указанной длины.",
        ge=0,
        examples=[2000]
    )
    group: bool = Field(
        default=True,
        description="Группировать ли результаты по category_path (true) или возвращать отдельные чанки (false)"
    )

    @field_validator("min_score", mode="before")
    @classmethod
    def default_min_score(cls, value):
        """null от клиента означает «значение по умолчанию», а не отключение."""
        return settings.search_min_score if value is None else value


class SearchCategoriesInput(BaseModel):
    query_text: str = Field(
        ...,
        description="Текст запроса для поиска по категориям",
        examples=["Документация по API"]
    )
    limit: int = Field(
        default=10,
        description="Макс. результатов (1-50)",
        ge=1,
        le=50,
        examples=[10]
    )
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


async def _search_categories_internal(
    query_text: str,
    limit: int = 10,
    fields: Optional[List[str]] = None,
    min_score: Optional[float] = settings.search_min_score,
    exact_phrases: Optional[List[str]] = None,
    phrase_mode: str = "any",
) -> Dict[str, Any]:
    """Внутренняя функция поиска категорий, без зависимости от FastAPI Request."""
    from app.api.categories import search_categories, _get_field_value

    if not query_text.strip():
        return {"success": True, "data": {"results": []}}

    if min_score is None:
        min_score = settings.search_min_score

    # Получаем категории
    categories = await search_categories(
        query_text, limit, min_score=min_score,
        exact_phrases=exact_phrases, phrase_mode=phrase_mode,
    )

    if not categories:
        return {"success": True, "data": {"results": []}}

    # Формируем результаты с учетом параметра fields
    if fields is None:
        # По умолчанию возвращаем только path и score
        results = [
            {"path": cat.category_path, "score": cat.score}
            for cat in categories
        ]
    else:
        # Формируем ответ с указанными полями
        results = [
            {field: _get_field_value(cat, field) for field in fields}
            for cat in categories
        ]
    
    return {"success": True, "data": {"results": results}}


async def _search_documents_internal(
    query_text: str,
    collection_name: Optional[str] = None,
    limit: int = 10,
    filter_criteria: Optional[Dict[str, Any]] = None,
    include_old_versions: bool = False,
    max_text_length: int = 0,
    group: bool = True,
    min_score: Optional[float] = settings.search_min_score,
    exact_phrases: Optional[List[str]] = None,
    phrase_mode: str = "any",
) -> Dict[str, Any]:
    """Внутренняя функция поиска документов, без зависимости от FastAPI Request.

    Args:
        group: Группировать ли результаты по category_path (true) или возвращать отдельные чанки (false)
        min_score: Минимальный score результата (0..1); меньшие отбрасываются
        exact_phrases: Фразы для точного поиска; совпадения идут первыми
        phrase_mode: "any" или "all" - режим exact-поиска
    """
    if min_score is None:
        min_score = settings.search_min_score

    # Подготавливаем метаданные для поиска
    result = await _prepare_search_metadata(query_text, filter_criteria, include_old_versions)
    if result["success"]:
        return result
    
    prepared = result["prepared"]
    client = get_client()
    
    # Если collection_name не указан, получаем список всех коллекций
    if not collection_name:
        try:
            collections_response = await client.get_collections()
            collection_names = [
                c.name for c in collections_response.collections
                if c.name != settings.category_collection
            ]
        except Exception as e:
            logger.warning(f"Failed to get collections: {e}")
            return {"success": False, "error": {"code": "collections_not_found", "message": str(e)}}
    else:
        collection_names = [collection_name]
    
    # Выполняем поиск по каждой коллекции и объединяем результаты
    phrases = normalize_phrases(exact_phrases)
    all_search_points = []
    exact_points = []
    for col_name in collection_names:
        try:
            points = await _perform_search(
                client, col_name, prepared["query_vector"], prepared["base_filter"],
                prepared["cat_ids"], limit * 2, prepared["category_path_filter"],
                min_score,
            )
            # Добавляем имя коллекции в метаданные каждой точки
            for point in points:
                point.payload['_collection_name'] = col_name
            all_search_points.extend(points)

            if phrases:
                points = await perform_exact_search(
                    client,
                    col_name,
                    prepared["query_vector"],
                    prepared["base_filter"],
                    phrases,
                    phrase_mode,
                    exact_candidate_limit(limit),
                )
                if prepared["category_path_filter"]:
                    points = [
                        p for p in points
                        if _matches_category_path(p.payload, prepared["category_path_filter"])
                    ]
                exact_points.extend(points)
        except ValueError as e:
            # Если collection_name был указан явно, возвращаем ошибку
            if collection_name:
                return {"success": False, "error": {"code": "collection_not_found", "message": str(e)}}
            logger.warning(f"Search in collection '{col_name}' failed: {e}")
            continue  # Пропускаем недоступные коллекции

    # Сортируем все результаты по релевантности и ограничиваем общее количество
    all_search_points.sort(key=lambda p: p.score, reverse=True)
    search_points = all_search_points[:limit * 2]

    # Exact-совпадения добавляем после среза семантических кандидатов
    if exact_points:
        search_points = merge_exact_points(search_points, exact_points)

    # Reranking чанков на основе схожести с вектором запроса
    if search_points and prepared["query_vector"] is not None:
        search_points = await rerank_by_query_similarity(search_points, prepared["query_vector"])

    # После reranking score — косинусное сходство, поэтому порог применяем
    # ещё раз на итоговых значениях. Exact-совпадения порог не отбрасывает.
    search_points = filter_by_min_score(search_points, min_score)

    # Exact-совпадения идут первыми, даже если их косинусный score ниже
    if phrases:
        search_points = sort_hybrid_points(search_points)
    
    # Обработка группировки/отдельных чанков
    from app.api.documents_utils import _prepare_search_context
    
    # Создаем временный request объект для передачи в функции группировки
    class TempRequest:
        def __init__(self):
            self.limit = limit
            self.max_text_length = max_text_length if max_text_length > 0 else None
            self.payload_fields = None
    
    temp_request = TempRequest()
    
    if group:
        # === Группировка результатов ===
        results = await _group_search_results(
            search_points=search_points,
            collection_names=collection_names,
            client=client,
            request=temp_request,
            payload_fields=None,
            query_vector=prepared["query_vector"],
        )
        # Преобразуем результаты в нужный формат
        formatted_results = []
        for r in results:
            formatted_results.append({
                "id": None,  # В группировке id не указан
                "score": r.score,
                "document": r.document,
                "payload": r.payload if hasattr(r, 'payload') and r.payload else None,
                "collection_name": r.collection_name,
                "category_path": r.category_path if hasattr(r, 'category_path') else None,
                "source_id": getattr(r, "source_id", None),
                "chunk_index": getattr(r, "chunk_index", None),
                "first_chunk_index": getattr(r, "first_chunk_index", None),
                "last_chunk_index": getattr(r, "last_chunk_index", None),
                "total_chunks": getattr(r, "total_chunks", None),
                "exact_match": getattr(r, "exact_match", None),
                "matched_phrases": getattr(r, "matched_phrases", None),
            })
        results = formatted_results
    else:
        # === Возврат отдельных чанков ===
        results = await _return_individual_chunks(
            search_points=search_points,
            collection_names=collection_names,
            request=temp_request,
            payload_fields=None,
            query_vector=prepared["query_vector"],
        )
        # Преобразуем результаты в нужный формат
        formatted_results = []
        for r in results:
            formatted_results.append({
                "id": None,
                "score": r.score,
                "document": r.document,
                "payload": r.payload if hasattr(r, 'payload') and r.payload else None,
                "collection_name": r.collection_name,
                "category_path": r.category_path if hasattr(r, 'category_path') else None,
                "source_id": getattr(r, "source_id", None),
                "chunk_index": getattr(r, "chunk_index", None),
                "first_chunk_index": getattr(r, "first_chunk_index", None),
                "last_chunk_index": getattr(r, "last_chunk_index", None),
                "total_chunks": getattr(r, "total_chunks", None),
                "exact_match": getattr(r, "exact_match", None),
                "matched_phrases": getattr(r, "matched_phrases", None),
            })
        results = formatted_results

    return {"success": True, "data": {"results": results}}


class GetCategoryHierarchyInput(BaseModel):
    collection_name: Optional[str] = Field(
        default=None,
        description="Имя коллекции (опционально). Если не указано, возвращает категории для всех коллекций.",
        examples=["documents"]
    )
    depth: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Количество уровней для получения (0 - только список коллекций без категорий, 1-10 - иерархия категорий).",
        examples=[1]
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Массив полных путей категорий (опционально). Если указаны, возвращает только коллекции с этими категориями.",
        examples=[["Документация / API", "Документация / руководства"]]
    )
    limit: Optional[int] = Field(
        default=5000,
        ge=1,
        description=(
            "Максимальное суммарное количество категорий в ответе. Коллекции "
            "обходятся по порядку, пока не исчерпан лимит. null - без ограничения."
        ),
        examples=[5000]
    )


async def _get_category_hierarchy_internal(
    collection_name: Optional[str] = None,
    depth: int = 1,
    categories: Optional[List[str]] = None,
    limit: Optional[int] = 5000,
) -> Dict[str, Any]:
    """Внутренняя функция получения иерархии категорий без зависимости от FastAPI Request."""
    from app.api.admin import _get_category_hierarchy_data

    try:
        result = await _get_category_hierarchy_data(
            collection_name=collection_name,
            depth=depth,
            categories=categories,
            limit=limit,
        )
        # Результат уже в нужном формате {"results": [...]}
        return {"success": True, "data": result}
    except ValueError as e:
        return {"success": False, "error": {"code": "collection_not_found", "message": str(e)}}
    except Exception as e:
        return {"success": False, "error": {"code": "hierarchy_failed", "message": str(e)}}


class GetGroupedTextInput(BaseModel):
    collection_name: Optional[str] = Field(
        default=None,
        description="Имя коллекции Qdrant. Если не указано, выборка выполняется по всем коллекциям, кроме служебной коллекции категорий.",
        examples=["documents"]
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description=(
            "Пути категорий для отбора. Учитываются точное совпадение и вложенные категории "
            "(например, 'A / B' включает 'A / B / C')."
        ),
        examples=[["Документация / API"]]
    )
    documents: Optional[List[str]] = Field(
        default=None,
        description="Список идентификаторов документов (source_id) для отбора.",
        examples=[["6ba7b810-9dad-11d1-80b4-00c04fd430c8"]]
    )
    include_old_versions: bool = Field(
        default=False,
        description="Включать старые версии документов (по умолчанию только актуальные)."
    )
    max_text_length: int = Field(
        default=0,
        description="Максимальная длина текста каждой группы (0 = без ограничений).",
        ge=0,
        examples=[2000]
    )
    limit: int = Field(
        default=20,
        description="Максимальное количество групп (документ + категория).",
        ge=1,
        le=1000,
        examples=[20]
    )


async def _get_grouped_text_internal(
    collection_name: Optional[str] = None,
    categories: Optional[List[str]] = None,
    documents: Optional[List[str]] = None,
    include_old_versions: bool = False,
    max_text_length: int = 0,
    limit: int = 20,
) -> Dict[str, Any]:
    """Внутренняя функция выборки сгруппированного текста для MCP."""
    try:
        results = await get_grouped_text(
            collection_name=collection_name,
            categories=categories,
            documents=documents,
            include_old_versions=include_old_versions,
            max_text_length=max_text_length,
            limit=limit,
        )
        return {"success": True, "data": {"results": results}}
    except ValueError as e:
        return {"success": False, "error": {"code": "collection_not_found", "message": str(e)}}
    except Exception as e:
        logger.exception("Grouped text retrieval failed")
        return {"success": False, "error": {"code": "grouped_text_failed", "message": str(e)}}