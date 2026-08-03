"""
Вспомогательные функции для работы с документами.
Содержит общие функции фильтрации, парсинга и обработки запросов.
"""
import logging
from typing import Optional, Dict, Any, List, Tuple

from qdrant_client.http import models as qdrant_models

from app.openapi_md.loader import load_openapi_md
from app.utils import basic_normalize, create_response
from app.models.common import DocumentsUploadRequest, DocumentSearchRequest

logger = logging.getLogger(__name__)

router = None  # Импортируется из основного модуля documents.py


def build_qdrant_filter(conditions: Optional[Dict[str, Any]]) -> Optional[qdrant_models.Filter]:
    """Строит фильтр для Qdrant на основе условий."""
    if not conditions:
        return None
    must_conditions = []
    for field, value in conditions.items():
        # category_path не включается в базовый фильтр, будет обработана отдельно
        if field == "category_path":
            continue
        must_conditions.append(
            qdrant_models.FieldCondition(
                key=field,
                match=qdrant_models.MatchValue(value=value),
            )
        )
    return qdrant_models.Filter(must=must_conditions)


def get_category_path_from_payload(payload: Dict[str, Any]) -> str:
    """Возвращает строковое представление category_path."""
    cat_path = payload.get("category_path")
    if isinstance(cat_path, list):
        return " / ".join(cat_path)
    return cat_path or ""


def get_all_chunks(client, collection_name: str, source_id: str, version: int) -> List[Dict]:
    """Получает все чанки документа по source_id и версии."""
    filter_ = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(key="source_id", match=qdrant_models.MatchValue(value=source_id)),
            qdrant_models.FieldCondition(key="version", match=qdrant_models.MatchValue(value=version)),
        ]
    )
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=filter_,
        limit=1000,  # достаточно для большинства документов
        with_payload=True,
    )
    return [p.payload for p in points]


def _matches_category_path(payload: Dict[str, Any], category_path_filter: str) -> bool:
    """
    Проверяет, соответствует ли category_path из payload фильтру.
    Включает точное совпадение и все вложенные категории.
    
    Args:
        payload: Метаданные точки Qdrant
        category_path_filter: Путь категории для фильтрации (например, "A / B / C")
    
    Returns:
        True, если category_path совпадает или является вложенной категорией
    """
    cat_path = payload.get("category_path")
    if not cat_path:
        return False
    
    # Приводим к строке, если это список
    if isinstance(cat_path, list):
        cat_path_str = " / ".join(cat_path)
    else:
        cat_path_str = str(cat_path)
    
    # Проверяем точное совпадение
    if cat_path_str == category_path_filter:
        return True
    
    # Проверяем, начинается ли category_path с фильтра + " / " (вложенная категория)
    prefix = category_path_filter + " / "
    return cat_path_str.startswith(prefix)


def _normalize_category_path(cat_path) -> str:
    """
    Нормализует category_path для сравнения.
    
    Args:
        cat_path: category_path (строка, список или другой тип)
    
    Returns:
        Нормализованная строка (без пробелов по краям)
    """
    if not cat_path:
        return ""
    if isinstance(cat_path, list):
        return " / ".join(cat_path).strip()
    elif not isinstance(cat_path, str):
        return str(cat_path).strip()
    return cat_path.strip()


async def _get_documents_upload_request_from_form_or_json(
    request,
    documents: Optional[str],
    collection_name: Optional[str],
) -> DocumentsUploadRequest:
    """
    Позволяет вводить параметры в Swagger как поля формы, но также принимает старый JSON body.

    Form mode:
    - documents: JSON строка (массив DocumentCreate)
    - collection_name: строка
    """
    if documents is not None or collection_name is not None:
        if not documents or not collection_name:
            raise Exception("Both 'documents' and 'collection_name' are required")
        try:
            docs_list = __import__('json').loads(documents)
        except Exception as e:
            raise Exception(f"Invalid 'documents' JSON: {e}")
        try:
            return DocumentsUploadRequest.model_validate(
                {"documents": docs_list, "collection_name": collection_name}
            )
        except Exception as e:
            raise Exception(str(e))

    try:
        data = await request.json()
    except Exception as e:
        raise Exception(f"Invalid JSON body: {e}")
    try:
        return DocumentsUploadRequest.model_validate(data)
    except Exception as e:
        raise Exception(str(e))

async def _prepare_search_context(
    request: DocumentSearchRequest,
):
    """
    Подготавливает общий контекст для поиска: клиент, вектор запроса, фильтр и категории.
    Возвращает кортеж (client, query_vector, base_filter, cat_ids) или response при пустом запросе.
    """
    from app.api.health import get_client
    from app.api.categories import search_categories
    from app.core.embeddings import encode_text

    client = get_client()
    query_text = basic_normalize(request.query_text)
    if not query_text:
        return create_response(
            success=True,
            data={
                "results": []
            }
        )

    categories = await search_categories(query_text, limit=10)
    cat_ids = [cat.id for cat in categories]

    query_vector = await encode_text(query_text, task_type="query")

    filter_conditions = request.filter or {}
    if not request.include_old_versions:
        filter_conditions["is_latest"] = True
    
    # Извлекаем category_path для последующей фильтрации
    category_path_filter = filter_conditions.pop("category_path", None)
    
    base_filter = build_qdrant_filter(filter_conditions)

    return client, query_vector, base_filter, cat_ids, category_path_filter
