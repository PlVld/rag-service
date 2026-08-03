"""
Эндпоинт загрузки документов.
Содержит эндпоинт /upload и функцию process_documents для обработки документов.
"""
import logging
import time
from typing import Optional, Dict, Any, List, Tuple

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from app.auth import verify_api_key

from app.openapi_md.loader import load_openapi_md
from app.utils import create_response, basic_normalize

from app.api.categories import update_category_hierarchy
from app.api.health import get_client
from app.api.documents_utils import (
    _get_documents_upload_request_from_form_or_json,
    build_qdrant_filter,
)
from app.chunking import ChunkerFactory
from app.core.config import settings
from app.models.common import DocumentsUploadRequest
from app.text_cleaning.pipeline import TextCleanerPipeline
from app.text_cleaning.heading_splitter import parse_heading_sections, build_full_category_path
from app.text_cleaning.normalizer import normalize_for_embedding
from app.utils import generate_uuid_from_parts, compute_doc_hash
from app.repository.qdrant_repository import QdrantBatchWriter
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from app.core.embeddings import encode_text

router = APIRouter(prefix="/v1/documents",
                   tags=["Documents"],
                   dependencies=[Depends(verify_api_key)],)
logger = logging.getLogger(__name__)

chunker = ChunkerFactory.create_chunker(
    chunker_type="langchain",
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)

text_cleaner = TextCleanerPipeline({
    'ignore_links': True,
    'ignore_images': False,
    'remove_punctuation': False,
    'remove_stopwords': False,
    'lowercase': True,
    'lemmatize': False,
})


async def process_documents(
        request: DocumentsUploadRequest,
        batch_writer: QdrantBatchWriter,
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Обрабатывает список документов для загрузки в векторную базу.

    Логика:
    1. Для каждого документа вычисляется source_id и хеш содержимого
    2. Проверяется наличие и актуальность существующей версии
    3. При необходимости создаются новые чанки и точки для вставки
    4. Обновляется иерархия категорий при наличии

    Возвращает:
        Список обновлённых source_id и идентификаторов созданных точек
    """
    start_time = time.time()
    logger.info(f"Starting document processing for {len(request.documents)} documents")
    total_points = []
    updated_source_ids = set()
    skipped_documents = []
    for doc in request.documents:
        if not doc.text:
            continue  # Пропускаем документы без текста

        original_text = doc.text
        source_format = doc.payload.get('source_format', 'text') if doc.payload else 'text'

        doc_version = doc.version or 1  # Версия по умолчанию — 1

        doc_metadata = doc.payload or {}  # Метаданные документа

        # Копируем важные поля DocumentCreate в doc_metadata (они могут не попадать из payload)
        if doc.category_path:
            doc_metadata["category_path"] = doc.category_path
        if doc.filename:
            doc_metadata.setdefault("original_filename", doc.filename)
        if doc.title:
            doc_metadata.setdefault("title", doc.title)

        # Генерация source_id, если не задан
        source_id = doc_metadata.get("source_id")
        if not source_id:
            category_path = doc_metadata.get("category_path")
            filename = doc_metadata.get("original_filename")
            title = doc_metadata.get("title")
            source_id = generate_uuid_from_parts([category_path, filename, title])
            doc_metadata["source_id"] = source_id  # Сохраняем для последующего использования

        # Добавляем технические метаданные
        doc_metadata["source_format"] = source_format
        doc_metadata["version"] = doc_version

        # Вычисляем хеш документа, если не передан
        doc_hash = doc_metadata.get("doc_hash")
        if doc_hash is None:
            doc_hash = compute_doc_hash(original_text)
            doc_metadata["doc_hash"] = doc_hash  # Кэшируем хеш

        # Создаем множество для отслеживания уникальных путей категорий, которые нужно обновить
        category_paths_to_update = set()

        # Добавляем путь категорий документа, если он указан
        category_path_list = doc_metadata.get("category_path")
        # Нормализуем: если строка, преобразуем в список (поддержка "cat1 / cat2" или "cat1/cat2")
        if isinstance(category_path_list, str):
            category_path_list = [c.strip() for c in category_path_list.replace(" / ", "/").split("/") if c.strip()]
            doc_metadata["category_path"] = category_path_list  # сохраняем список для дальнейшего использования
        if category_path_list and isinstance(category_path_list, list):
            category_paths_to_update.add(tuple(category_path_list))
            logger.debug(f"Added document category path: {category_path_list}")
            
        # Проверяем существование актуальной версии документа
        client = get_client()
        try:
            points, _ = client.scroll(
                collection_name=request.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(key="source_id", match=qdrant_models.MatchValue(value=source_id)),
                        qdrant_models.FieldCondition(key="is_latest", match=qdrant_models.MatchValue(value=True))
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            existing_points = list(points)
        except UnexpectedResponse as e:
            if "Not found: Collection" in str(e):
                existing_points = []  # Коллекция не существует — считаем, что документа нет
            else:
                raise  # Другие ошибки пропускаем

        # Сравниваем хеш с существующей версией
        existing_hash = None
        if existing_points:
            existing_payload = existing_points[0].payload
            existing_hash = existing_payload.get("doc_hash")
            
        # Если документ существует и хеш совпадает, пропускаем его обработку
        if existing_hash == doc_hash:
            logger.info(f"Document {source_id} already up-to-date, skipping.")
            continue  # Документ не изменился — пропускаем
        
        # Если документ обновляется, помечаем старые версии как неактуальные
        if existing_hash is not None:
            batch_writer.mark_old_versions_not_latest(request.collection_name, source_id, keep_version=doc_version)
            logger.info(f"Will mark old chunks of {source_id} as not latest")
            
        # Обновляем метаданные.
        # Сохраняем существующие level-поля из старой версии, если они есть
        if existing_points:
            for key, value in existing_points[0].payload.items():
                if key.startswith("category_level") and key not in doc_metadata:
                    doc_metadata[key] = value

        # Конвертируем в Markdown (до разбиения на секции по заголовкам)
        if source_format in ('markdown', 'code'):
            markdown_text = original_text
        else:
            markdown_text = text_cleaner.convert_to_markdown(original_text, source_format)

        if source_format == 'code':
            doc_metadata["content_type"] = "code"
        else:
            doc_metadata["content_type"] = "markdown"

        # Определяем категории документа (до заголовков)
        doc_category_list = []
        if category_path_list and isinstance(category_path_list, list):
            doc_category_list = category_path_list
        elif isinstance(category_path_list, str) and category_path_list:
            doc_category_list = [c.strip() for c in category_path_list.split(" / ") if c.strip()]

        doc_category_count = len(doc_category_list)
        doc_metadata["doc_category_count"] = doc_category_count

        # Разбиваем Markdown на секции по заголовкам (до нормализации)
        # Для кода не разбиваем по заголовкам
        if source_format == 'code':
            heading_sections = [type('Section', (), {
                'heading_hierarchy': [],
                'text': markdown_text,
                'level': 0,
            })()]
        else:
            heading_sections = parse_heading_sections(markdown_text)

        if not heading_sections:
            heading_sections = [type('Section', (), {
                'heading_hierarchy': [],
                'text': markdown_text,
                'level': 0,
            })()]

        logger.debug(
            f"Document {source_id}: split into {len(heading_sections)} heading sections, "
            f"doc_category_count={doc_category_count}"
        )

        # Чанкуем каждую секцию отдельно, строя полную иерархию категорий
        all_chunks = []  # list of (chunk_data, full_category_path)
        for section in heading_sections:
            section_metadata = doc_metadata.copy()

            # Строим полный путь: категории документа + иерархия заголовков
            full_cat_path = build_full_category_path(doc_category_list, section.heading_hierarchy)

            if full_cat_path:
                # Добавляем или обновляем информацию о категории
                section_metadata["category_path"] = " / ".join(full_cat_path)
                # Добавляем уровни категорий (category_level0, category_level1, ...)
                for i, category in enumerate(full_cat_path):
                    section_metadata[f"category_level{i}"] = category
                    # Проверяем, есть ли уже category_id_level в doc_metadata, чтобы избежать конфликта
                    if f"category_id_level{i}" not in section_metadata:
                        # Генерируем и добавляем ID для каждого уровня категории
                        full_path_for_id = " / ".join(full_cat_path[:i + 1])
                        category_id = generate_uuid_from_parts([full_path_for_id])
                        section_metadata[f"category_id_level{i}"] = category_id
                # Добавляем поле category_level - номер последнего уровня
                section_metadata["category_level"] = len(full_cat_path) - 1
                
                # Добавляем путь в множество для последующего обновления иерархии
                category_paths_to_update.add(tuple(full_cat_path))
            else:
                section_metadata["category_path"] = None
                section_metadata["category_level"] = -1

            section_chunks = chunker.chunk(section.text, section_metadata)
            for chunk_data in section_chunks:
                all_chunks.append((chunk_data, full_cat_path))

        if not all_chunks:
            logger.info(f"Document {source_id} produced no chunks, skipping.")
            skipped_documents.append({"source_id": source_id, "reason": "empty_after_cleaning"})
            continue

        # Собираем все нормализованные тексты для пакетной векторизации
        total_chunks = len(all_chunks)
        normalized_texts = []
        chunk_metadata_list = []
        chunk_indices = []
        chunk_cat_paths = []  # полные пути категорий для каждого чанка
        
        for idx, (chunk_data, full_cat_path) in enumerate(all_chunks):
            chunk_original = chunk_data["text"]  # это уже Markdown
            chunk_meta = chunk_data["metadata"].copy()

            # Полный category_path для нормализации (документ + заголовки)
            full_category_path_str = " / ".join(full_cat_path) if full_cat_path else doc_metadata.get("category_path", "")

            # Нормализация для эмбеддинга (Markdown -> чистый текст)
            chunk_normalized = normalize_for_embedding(
                chunk_original,
                source_format=doc_metadata["content_type"],
                category_path=full_category_path_str,
            )
            if not chunk_normalized:
                continue
                
            normalized_texts.append(chunk_normalized)
            chunk_metadata_list.append(chunk_meta)
            chunk_indices.append(idx)
            chunk_cat_paths.append(full_cat_path)
            
        # Пакетная векторизация
        if not normalized_texts:
            logger.warning(
                f"Document {source_id}: no chunks normalized after cleaning. "
                f"total_all_chunks={total_chunks}, all_chunks_count={len(all_chunks)}"
            )
            # Log samples for debugging
            for s_idx, (chunk_data, full_cat_path) in enumerate(all_chunks[:3]):
                sample_text = (chunk_data["text"] or "")[:500].replace("\n", " ")
                logger.debug(
                    f"Sample chunk {s_idx} (cat_path={' / '.join(full_cat_path) if full_cat_path else ''}): {sample_text}"
                )
            skipped_documents.append({"source_id": source_id, "reason": "no_normalized_chunks"})
            continue

        try:
            # encode_text для списка строк возвращает List[List[float]]
            vectors = await encode_text(normalized_texts, task_type="document")  # type: ignore
        except Exception as e:
            logger.exception(f"Encoding failed for document {source_id}")
            skipped_documents.append({"source_id": source_id, "reason": "encoding_failed", "error": str(e)})
            continue

        # Создаём точки с векторами
        points_for_doc = []
        for i, (idx, vector, chunk_meta) in enumerate(zip(chunk_indices, vectors, chunk_metadata_list)):
            chunk_meta["raw_text"] = all_chunks[idx][0]["text"]
            chunk_meta["normalized_text"] = normalized_texts[i]
            chunk_meta["chunk_index"] = idx
            chunk_meta["total_chunks"] = total_chunks
            chunk_meta["is_latest"] = True  # Флаг актуальной версии
            chunk_meta["doc_category_count"] = doc_category_count

            # Генерируем уникальный ID и вектор
            point_id = generate_uuid_from_parts([source_id, doc_version, idx])

            # Создаём точку для вставки
            point = qdrant_models.PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk_meta
            )
            points_for_doc.append(point)
            total_points.append(point)  # Для подсчёта результата

        # Добавляем все точки документа в батч
        for point in points_for_doc:
            batch_writer.add_point(request.collection_name, point)

        # Обновляем иерархию категорий для всех уникальных путей, найденных в документе и его чанках
        category_points = []  # Точки категорий для вставки
        if category_paths_to_update:
            logger.info(f"Document {source_id}: will update category paths: {list(category_paths_to_update)}")
        
        # Сначала сортируем пути по длине, чтобы обрабатывать более короткие пути (родительские) раньше
        sorted_paths = sorted(category_paths_to_update, key=lambda x: len(x))
        
        for cat_path_tuple in sorted_paths:
            cat_path_list = list(cat_path_tuple)
            cat_ids, level_payload, cat_points = await update_category_hierarchy(cat_path_list)
            logger.debug(f"Updated hierarchy for path {cat_path_list}, got {len(cat_points)} category points")
            
            if cat_ids:
                # Добавляем точки категорий в батч
                for point in cat_points:
                    batch_writer.add_point(settings.category_collection, point)
                    category_points.append(point)
                
                # Обновляем метаданные с информацией о категориях для использования в других частях процесса
                for i, category in enumerate(cat_path_list):
                    level_key = f"category_level{i}"
                    id_level_key = f"category_id_level{i}"
                    if level_key not in doc_metadata:
                        doc_metadata[level_key] = category
                    if id_level_key not in doc_metadata:
                        doc_metadata[id_level_key] = cat_ids[i]
                
                # Добавляем основную информацию о категории
                if "category_path" not in doc_metadata:
                    doc_metadata["category_path"] = " / ".join(cat_path_list)
                if "category_level" not in doc_metadata:
                    doc_metadata["category_level"] = len(cat_path_list) - 1

        # Фиксируем, что документ был обновлён
        updated_source_ids.add(source_id)

    elapsed = time.time() - start_time
    logger.info(f"Document processing completed in {elapsed:.3f}s: {len(updated_source_ids)} updated, {len(total_points)} points created")
    return list(updated_source_ids), [p.id for p in total_points], skipped_documents


@router.post("/upload",
             summary="Загрузка документов",
             description=load_openapi_md("documents_upload.md"),
             response_model_exclude_unset=True)
async def upload_documents(
    http_request: Request,
    documents: Optional[str] = Form(
        default=None,
        description="JSON-массив документов (строкой). Если не передан, можно отправить обычное JSON-тело запроса.",
    ),
    collection_name: Optional[str] = Form(
        default=None,
        description="Имя коллекции. Если не передано, можно отправить обычное JSON-тело запроса.",
    ),
):
    try:
        request = await _get_documents_upload_request_from_form_or_json(
            request=http_request,
            documents=documents,
            collection_name=collection_name,
        )
        batch_writer = QdrantBatchWriter()
        updated_source_ids, point_ids, skipped_docs = await process_documents(request, batch_writer)

        # Prepare standardized response using response utility
        if not point_ids:
            return create_response(
                success=True,
                data={
                    "uploaded": 0,
                    "ids": [],
                    "message": "Нет новых или изменившихся документов для загрузки"
                }
            )

        await batch_writer.commit()

        # type: ignore - response_data может содержать дополнительные ключи
        response_data: Dict[str, Any] = {
            "uploaded": len(point_ids),
            "ids": point_ids
        }

        # Добавляем информацию о пропущенных документах, если есть
        if skipped_docs:
            response_data["skipped"] = skipped_docs  # type: ignore

        return create_response(
            success=True,
            data=response_data
        )
    except Exception as e:
        logger.exception("Upload failed")
        return create_response(
            success=False,
            error_code="upload_failed",
            error_message=str(e)
        )
