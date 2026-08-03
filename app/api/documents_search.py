"""
Эндпоинт поиска документов.
Содержит эндпоинт /search и все вспомогательные функции для поиска.
"""
import logging
import json
from collections import defaultdict
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from fastapi import APIRouter, Depends, HTTPException
from app.auth import verify_api_key
from app.openapi_md.loader import load_openapi_md
from app.utils import create_response

from app.api.health import get_client
from app.api.documents_utils import (
    _prepare_search_context,
    build_qdrant_filter,
    _matches_category_path,
)
from app.core.config import settings
from app.models.common import DocumentSearchRequest, DocumentSearchResult, GroupedDocumentResult, VersionResult
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from app.core.embeddings import encode_text

router = APIRouter(prefix="/v1/documents",
                   tags=["Documents"],
                   dependencies=[Depends(verify_api_key)],)
logger = logging.getLogger(__name__)


async def _perform_search(client, collection_name: str, query_vector, base_filter, cat_ids, limit: int, category_path_filter: Optional[str] = None) -> List[
    qdrant_models.ScoredPoint]:
    """
    Выполняет поиск с использованием RRF при наличии категорий.
    
    Args:
        category_path_filter: Фильтр по category_path. Если задан, результаты будут отфильтрованы
                             на стороне сервера после поиска, чтобы включить только документы,
                             чей category_path начинается с указанного значения.
    """
    try:
        if not cat_ids:
            response = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                filter=base_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            # Применяем фильтр по category_path, если задан
            if category_path_filter:
                response.points = [
                    p for p in response.points
                    if _matches_category_path(p.payload, category_path_filter)
                ]
            return response.points

        # RRF с двумя prefetch
        prefetch_list = [
            qdrant_models.Prefetch(
                query=query_vector,
                filter=base_filter,
                limit=limit * 2
            ),
            qdrant_models.Prefetch(
                query=query_vector,
                filter=qdrant_models.Filter(
                    must=[
                             qdrant_models.FieldCondition(
                                 key="category_path_ids",
                                 match=qdrant_models.MatchAny(any=cat_ids)
                             )
                         ] + ([base_filter] if base_filter else [])
                    ),
                limit=limit * 2
            )
        ]
        response = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch_list,
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        # Применяем фильтр по category_path, если задан
        if category_path_filter:
            response.points = [
                p for p in response.points
                if _matches_category_path(p.payload, category_path_filter)
            ]
        return response.points
    except UnexpectedResponse as e:
        if "Not found: Collection" in str(e):
            logger.warning(f"Collection '{collection_name}' not found in Qdrant")
            raise ValueError(f"Collection '{collection_name}' not found") from e
        raise


@router.post("/search",
             summary="Поиск по документам",
             description=load_openapi_md("documents_search.md"))
async def search_documents(
    request: DocumentSearchRequest,
    group: Optional[bool] = None,
):
    """
    Выполняет семантический поиск по тексту в документах.
    
    Args:
        request: Запрос поиска с параметрами
        group: Группировать ли результаты по category_path. 
               Если не указано, используется значение из request.group (по умолчанию True).
               Параметр передается для совместимости с /grouped_search.
    """
    logger.info("Start search")
    
    try:
        # Подготавливаем контекст поиска
        from app.api.documents_utils import _prepare_search_context
        result = await _prepare_search_context(request)
        
        if isinstance(result, dict):
            return create_response(success=True, data={"results": []})  # Пустой результат при пустом запросе

        client, query_vector, base_filter, cat_ids, category_path_filter = result

        # Если collection_name не указан, получаем список всех коллекций
        if not request.collection_name:
            collections_response = client.get_collections()
            collection_names = [c.name for c in collections_response.collections
                                if c.name != settings.category_collection]
        else:
            collection_names = [request.collection_name]

        # Выполняем поиск по каждой коллекции и объединяем результаты
        all_search_points = []
        for col_name in collection_names:
            try:
                points = await _perform_search(client, col_name, query_vector, base_filter, cat_ids, request.limit * 2, category_path_filter)
                # Добавляем имя коллекции в метаданные каждой точки
                for point in points:
                    point.payload['_collection_name'] = col_name
                all_search_points.extend(points)
            except ValueError as e:
                # Если collection_name был указан явно, возвращаем ошибку
                if request.collection_name:
                    raise
                logger.warning(f"Search in collection '{col_name}' failed: {e}")
                continue  # Пропускаем недоступные коллекции

        # Сортируем все результаты по релевантности и ограничиваем общее количество
        all_search_points.sort(key=lambda p: p.score, reverse=True)
        search_points = all_search_points[:request.limit * 2]
        
        # Reranking чанков на основе схожести с вектором запроса
        if search_points and query_vector is not None:
            search_points = await rerank_by_query_similarity(search_points, query_vector)
        
        payload_fields_list = request.payload_fields
        # Используем переданный параметр group с приоритетом, иначе значение из request
        use_grouping = group if group is not None else request.group
        if use_grouping:
            # === Группировка результатов ===
            results = await _group_search_results(
                search_points=search_points,
                collection_names=collection_names,
                client=client,
                request=request,
                payload_fields=payload_fields_list,
                query_vector=query_vector,
            )
        else:
            # === Возврат отдельных чанков ===
            results = await _return_individual_chunks(
                search_points=search_points,
                collection_names=collection_names,
                request=request,
                payload_fields=payload_fields_list,
                query_vector=query_vector,
            )

        return create_response(success=True, data={"results": results})

    except ValueError as e:
        return create_response(success=False, error_code="collection_not_found", error_message=str(e))
    except Exception as e:
        logger.exception("Search failed")
        return create_response(success=False, error_code="search_failed", error_message=str(e))


async def _group_search_results(
    search_points: List[qdrant_models.ScoredPoint],
    collection_names: List[str],
    client: Any,
    request: DocumentSearchRequest,
    payload_fields: Optional[List[str]],
    query_vector: Optional[List[float]] = None,
) -> List[DocumentSearchResult]:
    """
    Группирует результаты по category_path и возвращает объединенные документы.
    Проводит реранкинг после группировки.
    
    Args:
        search_points: Результаты поиска из Qdrant
        collection_names: Список коллекций для поиска
        client: Клиент Qdrant
        request: Запрос поиска
        payload_fields: Список полей payload для возврата
        query_vector: Вектор запроса для реранкинга (опционально)
    
    Returns:
        Список DocumentSearchResult с группированными результатами
    """
    from app.api.documents_utils import get_all_chunks, _normalize_category_path
    
    # Собираем уникальные category_path из результатов
    unique_category_paths = set()
    for point in search_points:
        category_path = point.payload.get("category_path")
        if category_path:
            # Преобразуем в строку, если это список
            if isinstance(category_path, list):
                category_path = " / ".join(category_path)
            elif not isinstance(category_path, str):
                category_path = str(category_path)
            unique_category_paths.add(category_path)

    # Группировка по category_path
    groups: Dict[str, List[qdrant_models.ScoredPoint]] = {}
    for point in search_points:
        category_path = point.payload.get("category_path")
        if category_path:
            # Преобразуем в строку, если это список
            if isinstance(category_path, list):
                category_path = " / ".join(category_path)
            elif not isinstance(category_path, str):
                category_path = str(category_path)
            # Нормализуем путь (убираем лишние пробелы)
            normalized_path = category_path.strip()
            groups.setdefault(normalized_path, []).append(point)

    # Собираем результаты для каждой группы
    group_results = []
    # Словарь для хранения имени коллекции для каждого source_id
    source_id_collections: Dict[str, str] = {}
    
    for category_path, group_points in groups.items():
        # Создаем единый результат для категории.
        # Берем payload первого (наиболее релевантного) чанка как основу
        sample_point = group_points[0]
        sample_payload = sample_point.payload
        source_id = sample_payload.get("source_id")
        version = sample_payload.get("version", 1)

        # Если для этого source_id еще не найдена коллекция, ищем её
        if source_id not in source_id_collections:
            for col_name in collection_names:
                try:
                    chunks = get_all_chunks(client, col_name, source_id, version)
                    if chunks:
                        source_id_collections[source_id] = col_name
                        break
                except Exception as e:
                    logger.debug(f"Document {source_id} not found in collection {col_name}: {e}")
                    continue
        
        col_name = source_id_collections.get(source_id)
        if not col_name:
            continue

        # Получаем чанки из найденной коллекции
        try:
            all_chunks = get_all_chunks(client, col_name, source_id, version)
        except Exception as e:
            logger.debug(f"Document {source_id} not found in collection {col_name}: {e}")
            continue

        if not all_chunks:
            continue

        # Сортируем все чанки документа по chunk_index
        sorted_all_chunks = sorted(all_chunks, key=lambda x: x.get('chunk_index', 0))

        # Собираем chunk_index из релевантных чанков этой категории
        relevant_indices = [p.payload.get("chunk_index", 0) for p in group_points]
        min_idx = min(relevant_indices)
        max_idx = max(relevant_indices)

        # Определяем границы контекста с учетом соседних чанков
        context_start = max(0, min_idx - 1)
        context_end = min(len(sorted_all_chunks) - 1, max_idx + 1)

        # Фильтруем чанки в диапазоне, оставляя только те, которые принадлежат к текущей category_path
        category_chunks_in_context = []
        for chunk in sorted_all_chunks[context_start:context_end + 1]:
            chunk_cat_path = chunk.get("category_path")
            if chunk_cat_path:
                chunk_cat_path = _normalize_category_path(chunk_cat_path)
                if chunk_cat_path == category_path:
                    category_chunks_in_context.append(chunk)

        if not category_chunks_in_context:
            # Если после фильтрации по category_path чанков не осталось, используем только релевантные чанки из группы
            for p in group_points:
                p_cat_path = p.payload.get("category_path")
                if p_cat_path:
                    p_cat_path = _normalize_category_path(p_cat_path)
                    if p_cat_path == category_path:
                        category_chunks_in_context.append(p.payload)

        # Сортируем отфильтрованные чанки по индексу
        category_chunks_in_context.sort(key=lambda x: x.get('chunk_index', 0))

        # Определяем итоговые first и last индексы из отфильтрованного списка
        first_chunk_index = category_chunks_in_context[0].get("chunk_index", 0)
        last_chunk_index = category_chunks_in_context[-1].get("chunk_index", 0)

        # Собираем текст
        full_raw_text = "\n".join(
            chunk.get("raw_text", "")
            for chunk in category_chunks_in_context
        )

        # Применяем ограничение длины текста, если задано
        max_len = request.max_text_length
        if max_len is not None and max_len > 0:
            full_raw_text = full_raw_text[:max_len]

        # Используем максимальный score из группы как итоговый score для реранкинга
        group_max_score = max(p.score for p in group_points)

        # Формируем payload для реранкинга (только нужные поля)
        group_payload = {}
        if payload_fields:
            for field in payload_fields:
                if field in sample_payload:
                    group_payload[field] = sample_payload[field]
        else:
            # Если payload_fields не указан, не возвращаем payload вообще
            group_payload = None

        group_results.append({
            "score": group_max_score,
            "document": full_raw_text,
            "payload": group_payload,
            "collection_name": col_name,
            "category_path": category_path,
            "source_id": source_id,
            "first_chunk_index": first_chunk_index,
            "last_chunk_index": last_chunk_index,
            "version": version,
            "raw_payload": sample_payload,
            "category_chunks": category_chunks_in_context,  # Добавляем чанки для сбора payload
        })

    # === Реранкинг после группировки ===
    # Создаем векторы для сгруппированных результатов (используем max score как "вектор" для сортировки)
    if group_results and query_vector is not None:
        # Для реранкинга групп используем max score как признак
        # В реальности можно использовать средний score или другой агрегат
        group_scores = [g["score"] for g in group_results]
        
        # Сортируем группы по score (убывание)
        group_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Обновляем score с учетом позиции (RRF-style)
        for i, group in enumerate(group_results):
            # Простая формула: score * 0.7 + position_bonus * 0.3
            position_bonus = 1.0 / (i + 1)  # RRF-style
            group["score"] = group["score"] * 0.7 + position_bonus * 0.3

    # Ограничиваем результаты
    group_results = group_results[:request.limit]

    # Преобразуаем в финальный формат DocumentSearchResult
    final_results = []
    for group in group_results:
        # Если payload_fields не указан, payload не добавляем вообще
        result_kwargs = {
            "category_path": group["category_path"],
            "score": group["score"],
            "document": group["document"],
            "collection_name": group["collection_name"],
        }
        # Добавляем payload только если payload_fields указан и не пустой
        if payload_fields:
            # Собираем значения полей из всех чанков в группе
            result_payload = {}
            for field in payload_fields:
                # Собираем все значения этого поля из всех чанков в группе
                all_values = []
                for chunk in group["category_chunks"]:
                    if field in chunk:
                        value = chunk[field]
                        # Проверяем, есть ли это значение уже в списке
                        if value not in all_values:
                            all_values.append(value)
                
                # Если нашли значения
                if all_values:
                    # Если значение одно, возвращаем его как скаляр, иначе как массив
                    if len(all_values) == 1:
                        result_payload[field] = all_values[0]
                    else:
                        result_payload[field] = all_values
            
            # Добавляем payload только если он не пустой
            if result_payload:
                result_kwargs["payload"] = result_payload
        
        final_results.append(DocumentSearchResult(**result_kwargs))

    logger.info(f"DEBUG: _group_search_results возвращает {len(final_results)} результатов")
    return final_results


async def _return_individual_chunks(
    search_points: List[qdrant_models.ScoredPoint],
    collection_names: List[str],
    request: DocumentSearchRequest,
    payload_fields: Optional[List[str]],
    query_vector: Optional[List[float]] = None,
) -> List[DocumentSearchResult]:
    """
    Возвращает отдельные чанки без группировки.
    
    Args:
        search_points: Результаты поиска из Qdrant
        collection_names: Список коллекций для поиска
        request: Запрос поиска
        payload_fields: Список полей payload для возврата
        query_vector: Вектор запроса (не используется, но для совместимости)
    
    Returns:
        Список DocumentSearchResult с отдельными чанками
    """
    # Создаем словарь для хранения уникальных результатов по category_path
    unique_results = {}
    for point in search_points:
        category_path = point.payload.get("category_path")
        if category_path:
            # Преобразуем в строку, если это список
            if isinstance(category_path, list):
                category_path = " / ".join(category_path)
            elif not isinstance(category_path, str):
                category_path = str(category_path)
            
            if category_path not in unique_results:
                # Извлекаем имя коллекции из метаданных точки
                col_name = point.payload.get('_collection_name', '')
                
                # Формируем kwargs для создания DocumentSearchResult
                result_kwargs = {
                    "category_path": category_path,
                    "score": point.score,
                    "document": point.payload.get("raw_text", ""),
                    "collection_name": col_name,
                }
                # Добавляем payload только если payload_fields указан и не пустой
                if payload_fields:
                    # Формируем payload
                    result_payload = {}
                    for field in payload_fields:
                        if field in point.payload:
                            result_payload[field] = point.payload[field]
                        else:
                            # Поле не найдено - просто пропускаем
                            pass
                    if result_payload:
                        result_kwargs["payload"] = result_payload
                
                unique_results[category_path] = DocumentSearchResult(**result_kwargs)

    results = list(unique_results.values())

    # Применяем ограничение длины текста, если задано
    max_len = request.max_text_length
    if max_len is not None and max_len > 0:
        for result in results:
            if len(result.document) > max_len:
                result.document = result.document[:max_len]

    return results


async def rerank_by_query_similarity(
    points: List[qdrant_models.ScoredPoint],
    query_vector: List[float],
) -> List[qdrant_models.ScoredPoint]:
    """
    Пересчитывает score для каждого чанка на основе косинусного сходства с вектором запроса.
    
    Это улучшает качество ранжирования, так как использует точное векторное сходство вместо
    приближенных метрик (RRF или фьюзия).
    
    Args:
        points: Чанки из Qdrant с исходным score
        query_vector: Вектор запроса (нормализованный)
    
    Returns:
        Чанки с пересчитанными score (отсортированы по убыванию сходства)
    """
    try:
        # Проверяем наличие вектора до логирования
        has_query_vector = False
        if query_vector is not None:
            has_query_vector = len(query_vector) > 0
        logger.info(f"DEBUG rerank: points={len(points) if points else 0}, has_query_vector={has_query_vector}")
        
        if not points or query_vector is None:
            logger.warning("Reranking skipped: no query_vector")
            return points
        if len(query_vector) == 0:
            logger.warning("Reranking skipped: query_vector is empty")
            return points
        
        # Логируем содержимое вектора для диагностики
        sample_values = query_vector[:3] if len(query_vector) >= 3 else query_vector
        logger.info(f"DEBUG rerank: query_vector sample values: {sample_values}, all_numeric={all(isinstance(v, (int, float)) for v in query_vector)}")
        
        # Логируем точки для диагностики
        sample_point_vectors = [p.vector for p in points[:3] if p.vector is not None]
        sample_point_none_vectors = [p.id for p in points[:3] if p.vector is None]
        logger.info(f"DEBUG rerank: points vectors: non-None count={len(sample_point_vectors)}, None vectors point_ids={sample_point_none_vectors}")
        
        # Извлекаем векторы в numpy массив (оптимизированная операция)
        vectors = np.array([p.vector for p in points])  # shape: (n_points, embedding_dim)
        query = np.array(query_vector)  # shape: (embedding_dim,)
        
        # Нормализуем векторы (если они не нормализованы)
        # Qdrant обычно использует нормализованные векторы (cosine similarity)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        query_norm = query / np.linalg.norm(query)
        
        # Вычисляем косинусное сходство (быстрая матричная операция)
        similarities = np.dot(vectors_norm, query_norm)  # shape: (n_points,)
        
        # Создаем новые точки с пересчитанным score
        reranked_points = []
        for i, point in enumerate(points):
            new_score = float(similarities[i])
            
            # Создаем новую точку с обновленным score
            reranked_points.append(
                qdrant_models.ScoredPoint(
                    id=point.id,
                    version=point.version,
                    score=new_score,  # ✅ Новый score = косинусное сходство
                    payload=point.payload,
                    vector=point.vector
                )
            )
        
        # Сортируем по новому score и возвращаем
        reranked_points.sort(key=lambda p: p.score, reverse=True)
        return reranked_points
        
    except Exception as e:
        logger.warning(f"Failed to rerank points: {e}. Using original scores.")
        return points


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
