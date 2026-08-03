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
from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field

from app.api.categories import search_categories
from app.api.health import get_client
from app.api.documents import build_qdrant_filter, _perform_search, get_all_chunks, _matches_category_path
from app.api.documents_search import _group_search_results, _return_individual_chunks, rerank_by_query_similarity
from app.core.config import settings
from app.core.embeddings import encode_text
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
    fields: Optional[List[str]] = Field(
        default=None,
        description=(
            "Список полей для возврата. Если не указан, возвращаются только `path` и `score`. "
            "Доступные поля: `id`, `score`, `category_name`, `category_path`, `categories`, "
            "`category_level`, `category_id`, `levels`, `id_levels`."
        ),
        examples=[["path", "score"]]
    )


async def _search_categories_internal(
    query_text: str,
    limit: int = 10,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Внутренняя функция поиска категорий, без зависимости от FastAPI Request."""
    from app.api.categories import search_categories, _get_field_value
    
    if not query_text.strip():
        return {"success": True, "data": {"results": []}}
    
    # Получаем категории
    categories = await search_categories(query_text, limit)
    
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
) -> Dict[str, Any]:
    """Внутренняя функция поиска документов, без зависимости от FastAPI Request.
    
    Args:
        group: Группировать ли результаты по category_path (true) или возвращать отдельные чанки (false)
    """
    # Подготавливаем метаданные для поиска
    result = await _prepare_search_metadata(query_text, filter_criteria, include_old_versions)
    if result["success"]:
        return result
    
    prepared = result["prepared"]
    client = get_client()
    
    # Если collection_name не указан, получаем список всех коллекций
    if not collection_name:
        try:
            collections_response = client.get_collections()
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
    all_search_points = []
    for col_name in collection_names:
        try:
            points = await _perform_search(
                client, col_name, prepared["query_vector"], prepared["base_filter"],
                prepared["cat_ids"], limit * 2, prepared["category_path_filter"]
            )
            # Добавляем имя коллекции в метаданные каждой точки
            for point in points:
                point.payload['_collection_name'] = col_name
            all_search_points.extend(points)
        except ValueError as e:
            # Если collection_name был указан явно, возвращаем ошибку
            if collection_name:
                return {"success": False, "error": {"code": "collection_not_found", "message": str(e)}}
            logger.warning(f"Search in collection '{col_name}' failed: {e}")
            continue  # Пропускаем недоступные коллекции
    
    # Сортируем все результаты по релевантности и ограничиваем общее количество
    all_search_points.sort(key=lambda p: p.score, reverse=True)
    search_points = all_search_points[:limit * 2]
    
    # Reranking чанков на основе схожести с вектором запроса
    if search_points and prepared["query_vector"] is not None:
        search_points = await rerank_by_query_similarity(search_points, prepared["query_vector"])
    
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


async def _get_category_hierarchy_internal(
    collection_name: Optional[str] = None,
    depth: int = 1,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Внутренняя функция получения иерархии категорий без зависимости от FastAPI Request."""
    from app.api.admin import _get_category_hierarchy_data
    
    try:
        result = await _get_category_hierarchy_data(
            collection_name=collection_name,
            depth=depth,
            categories=categories
        )
        # Результат уже в нужном формате {"results": [...]}
        return {"success": True, "data": result}
    except ValueError as e:
        return {"success": False, "error": {"code": "collection_not_found", "message": str(e)}}
    except Exception as e:
        return {"success": False, "error": {"code": "hierarchy_failed", "message": str(e)}}
