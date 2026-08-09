import asyncio
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from qdrant_client.http import models as qdrant_models

import logging
logger = logging.getLogger(__name__)

def generate_uuid_from_parts(parts: List[Any], namespace: uuid.UUID = uuid.NAMESPACE_DNS) -> str:
    def flatten(item):
        if item is None:
            return []
        if isinstance(item, (list, tuple)):
            result = []
            for sub in item:
                result.extend(flatten(sub))
            return result
        else:
            return [str(item)]

    flat = flatten(parts)
    if not flat:
        return str(uuid.uuid4())
    unique_string = "|".join(flat)
    # Используем uuid5 для детерминированного ID на основе строки
    return str(uuid.uuid5(namespace, unique_string))

def compute_doc_hash(text: str) -> str:
    """Вычисляет MD5-хеш всего текста документа."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def basic_normalize(text: str) -> str:
    """Простая нормализация для поискового запроса."""
    if not text:
        return ""
    return ' '.join(text.lower().split())


def create_response(
    success: bool,
    data: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    version: str = "1.0.0"
) -> Dict[str, Any]:
    """
    Создает стандартизированный ответ для API.
    
    Args:
        success: Флаг успешности операции
        data: Данные результата (или None при ошибке)
        error_code: Код ошибки (или None при успехе)
        error_message: Сообщение об ошибке (или None при успехе)
        version: Версия формата ответа
    
    Returns:
        Стандартизированный ответ в формате JSON
    """
    return {
        "success": success,
        "data": data,
        "error": {
            "code": error_code,
            "message": error_message,
            "details": None
        } if error_code or error_message else None,
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": version
        }
    }

def create_file_upload_response(
    status: str,
    source_id: Optional[str] = None,
    version: Optional[int] = None,
    is_latest: bool = True,
    content_hash: Optional[str] = None,
    uploaded_chunks: int = 0,
    chunk_ids: list = None,
    original_filename: Optional[str] = None,
    collection_name: Optional[str] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Создает ответ для операции загрузки файла.
    
    Args:
        status: Статус операции (created, updated, skipped, error)
        source_id: Идентификатор источника
        version: Версия документа
        is_latest: Является ли версия актуальной
        content_hash: Хеш содержимого
        uploaded_chunks: Количество загруженных чанков
        chunk_ids: Список идентификаторов чанков
        original_filename: Оригинальное имя файла
        collection_name: Название коллекции
        error_code: Код ошибки
        error_message: Сообщение об ошибке
    
    Returns:
        Стандартизированный ответ для загрузки файла
    """
    chunk_ids = chunk_ids or []
    
    data = {
        "source_id": source_id,
        "version": version,
        "status": status,
        "is_latest": is_latest,
        "content_hash": content_hash,
        "uploaded_chunks": uploaded_chunks,
        "chunk_ids": chunk_ids,
        "original_filename": original_filename,
        "collection_name": collection_name
    } if not error_code and not error_message else None
    
    # Убираем None значения из data
    if data:
        data = {k: v for k, v in data.items() if v is not None}
    
    return create_response(
        success=not (error_code or error_message) and status != "error",
        data=data,
        error_code=error_code,
        error_message=error_message
    )

def create_batch_upload_response(
    results: list,
    total: int
) -> Dict[str, Any]:
    """
    Создает ответ для пакетной загрузки файлов.
    
    Args:
        results: Список результатов загрузки каждого файла
        total: Общее количество файлов
    
    Returns:
        Стандартизированный ответ для пакетной загрузки
    """
    processed = len([r for r in results if r["success"]])
    failed = len([r for r in results if not r["success"]])
    
    return {
        "success": True,
        "data": {
            "results": results,
            "total": total,
            "processed": processed,
            "failed": failed
        },
        "error": None,
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0"
        }
    }


async def scroll_all_pages(
    client: Any,
    collection_name: str,
    scroll_filter: Optional[Any],
    limit: int,
    with_payload: Union[bool, List[str]] = True,
    with_vectors: Union[bool, List[str]] = False,
) -> List[qdrant_models.PointStruct]:
    """
    Асинхронно скроллирует все точки из Qdrant-коллекции, обрабатывая пагинацию.

    Внутренний цикл вызываet ``client.scroll`` пока ``next_page_offset`` не станет ``None``,
    тем самым гарантируя, что документы больше лимита не будут молча обрезаны.

    Args:
        client: Экземпляр асинхронного клиента QdrantClient.
        collection_name: Название коллекции.
        scroll_filter: Фильтр для выборки точек (Qdrant Filter).
        limit: Количество точек на один запрос (по умолчанию 1000).
        with_payload:True / список имён полей / False.
        with_vectors: True / список имён векторов / False.

    Returns:
        Полный список точек, найденных по фильтру.
    """
    all_points: List[qdrant_models.PointStruct] = []
    offset = None
    total_fetched = 0

    while True:
        points, next_offset = await client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        all_points.extend(points)
        total_fetched += len(points)
        if next_offset is None:
            break
        offset = next_offset

    if total_fetched > limit:
        logger.debug(
            "scroll_all_pages: %s — %d точек собрано за %d запросов (limit=%d)",
            collection_name,
            total_fetched,
            total_fetched // limit + (1 if total_fetched % limit else 0),
            limit,
        )

    return all_points