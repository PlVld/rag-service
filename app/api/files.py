import os
import uuid
import json
import aiofiles
import logging
import time
import hashlib
import mimetypes
from datetime import datetime, timezone
from typing import Optional, List, Any, Tuple, Union, Dict
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import PlainTextResponse
from app.auth import verify_api_key
from app.openapi_md.loader import load_openapi_md

from app.utils import create_file_upload_response, create_batch_upload_response, create_response

from app.api.documents import process_documents
from app.models.common import DocumentsUploadRequest, DocumentCreate
from app.core.config import settings
from app.utils import generate_uuid_from_parts
from app.api.health import get_client
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from app.repository.qdrant_repository import QdrantBatchWriter

router = APIRouter(prefix="/v1/files",
                   tags=["Files"],
                   dependencies=[Depends(verify_api_key)],)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Разрешённые расширения файлов
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".markdown"}

# Карта MIME-type → расширения для валидации
MIME_TYPE_MAP = {
    "text/plain": [".txt", ".md", ".markdown"],
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "application/msword": [".doc"],
    "text/html": [".html", ".htm"],
    "text/markdown": [".md", ".markdown"],
}


def validate_mime_type_and_extension(filename: str, detected_mime: str) -> None:
    """
    Валидирует MIME-type и расширение файла.

    Args:
        filename: Имя загружаемого файла.
        detected_mime: Определённый MIME-type.

    Raises:
        HTTPException: Если тип файла не поддерживается или расширение не соответствует MIME-type.
    """
    file_ext = os.path.splitext(filename)[1].lower()

    # Проверяем расширение
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{file_ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Проверяем, что detected MIME-type входит в разрешённые
    allowed_mimes = set(MIME_TYPE_MAP.keys())
    if detected_mime not in allowed_mimes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported MIME type '{detected_mime}'. Allowed: {', '.join(sorted(allowed_mimes))}"
        )

    # Проверяем соответствие расширения MIME-type
    expected_extensions = MIME_TYPE_MAP.get(detected_mime, [])
    if file_ext not in expected_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{file_ext}' does not match detected MIME type '{detected_mime}'"
        )


def detect_mime_type(file_path: str) -> str:
    """
    Определяет MIME-type файла по содержимому.

    Args:
        file_path: Путь к файлу.

    Returns:
        str: Определённый MIME-type.
    """
    # Пробуем python-magic (более надёжно)
    try:
        import magic
    except ImportError:
        magic = None
    
    if magic:
        try:
            mime = magic.from_file(file_path, mime=True)
            if mime:
                return mime  # type: ignore
        except (AttributeError, TypeError, OSError):
            pass  # magic не сработал, используем fallback

    # Fallback: определяем по расширению и сигнатурам
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        # Читаем первые байты для определения типа
        with open(file_path, "rb") as f:
            header = f.read(8)
            if header[:4] == b'%PDF':
                return "application/pdf"
            elif header[:4] == b'PK\x03\x04':
                # DOCX/ZIP формат — проверяем наличие word/document.xml
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif header[:4] == b'\xd0\xcf\x11\xe0':
                # OLE формат (старый .doc)
                return "application/msword"
            else:
                return "text/plain"
    return mime_type


async def extract_text_from_file(file_path: str) -> str:
    """
    Асинхронно извлекает текст из файла в зависимости от его формата.

    Args:
        file_path (str): Путь к файлу для чтения.

    Returns:
        str: Содержимое файла в виде строки.

    Raises:
        IOError: Если файл не может быть прочитан.
    """
    logger.info(f"Extracting text from file: {file_path}")
    try:
        # Определяем формат файла по расширению
        file_ext = os.path.splitext(file_path)[1].lower()

        # Пробуем Docling если включен в настройках
        if settings.use_docling:
            try:
                logger.info("Attempting Docling conversion")
                from app.text_cleaning.docling_cleaner import DoclingCleaner
                docling_cleaner = DoclingCleaner(
                    do_ocr=settings.docling_do_ocr,
                    ocr_engine=settings.docling_ocr_engine,
                    image_description_model=settings.docling_image_description_model or None,
                    images_scale=settings.docling_images_scale,
                )
                text_content = docling_cleaner.clean(file_path)
                if text_content and text_content.strip():
                    logger.info(f"Docling conversion successful, length: {len(text_content)} characters")
                    return text_content
                else:
                    logger.warning("Docling produced empty content, falling back to legacy extractors")
            except Exception as docling_e:
                logger.warning(f"Docling conversion failed, falling back to legacy extractors: {docling_e}")

        # Legacy extractors (fallback)
        if file_ext == '.pdf':
            # Для PDF файлов используем специальный обработчик
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()
                logger.info(f"Read PDF file, size: {len(content)} bytes")
                # Создаем экземпляр PDFCleaner и извлекаем текст
                from app.text_cleaning.pdf_cleaner import PDFCleaner
                cleaner = PDFCleaner()
                text_content = cleaner.clean(content.decode('utf-8', errors='replace'))
                logger.info(f"Successfully extracted text from PDF, length: {len(text_content)} characters")
                return text_content
        elif file_ext in ['.docx', '.doc']:
            # Для DOCX и DOC файлов используем DOCXCleaner
            logger.info(f"Processing DOC/DOCX file: {file_path}")
            from app.text_cleaning.doc_cleaner import DOCXCleaner
            cleaner = DOCXCleaner()
            text_content = cleaner.clean(file_path)
            logger.info(f"Successfully extracted text from DOC/DOCX, length: {len(text_content)} characters")
            return text_content
        else:
            # Для других форматов читаем как текст
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = await f.read()
                logger.info(f"Successfully extracted text, length: {len(content)} characters")
                return content

    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {str(e)}")
        raise


async def convert_file_to_markdown_raw(
    file: UploadFile,
) -> dict:
    """
    Конвертирует загруженный файл в Markdown (raw_text) без чанкинга и без записи в Qdrant.
    Возвращает структуру с исходным именем, detected_mime, source_format и markdown_text.
    """
    temp_path = None
    try:
        temp_path, file_size, _ = await save_file_and_compute_hash(file)
        detected_mime = detect_mime_type(temp_path)
        validate_mime_type_and_extension(file.filename, detected_mime)
        file_ext = os.path.splitext(file.filename)[1].lower()

        markdown_text = None

        # Пробуем Docling если включен
        if settings.use_docling:
            try:
                logger.info("Attempting Docling conversion for markdown preview")
                from app.text_cleaning.docling_cleaner import DoclingCleaner
                docling_cleaner = DoclingCleaner(
                    do_ocr=settings.docling_do_ocr,
                    ocr_engine=settings.docling_ocr_engine,
                    image_description_model=settings.docling_image_description_model or None,
                    images_scale=settings.docling_images_scale,
                )
                markdown_text = docling_cleaner.clean(temp_path)
                if markdown_text and markdown_text.strip():
                    logger.info(f"Docling conversion successful, length: {len(markdown_text)}")
                else:
                    logger.warning("Docling produced empty content, falling back to legacy converters")
                    markdown_text = None
            except Exception as docling_e:
                logger.warning(f"Docling conversion failed, falling back: {docling_e}")
                markdown_text = None

        # Legacy fallback если Docling отключен или не сработал
        source_format = "text"  # по умолчанию
        if markdown_text is None:
            if file_ext == ".pdf":
                source_format = "pdf"
                async with aiofiles.open(temp_path, "rb") as f:
                    content = await f.read()
                from app.text_cleaning.pdf_cleaner import PDFCleaner
                markdown_text = PDFCleaner().clean(content.decode('utf-8', errors='replace'))
            elif file_ext in [".docx", ".doc"]:
                source_format = "docx"
                from app.text_cleaning.doc_cleaner import DOCXCleaner
                markdown_text = DOCXCleaner().clean(temp_path)
            elif file_ext in [".md", ".markdown"]:
                source_format = "markdown"
                async with aiofiles.open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                    markdown_text = await f.read()
            elif file_ext in [".html", ".htm"]:
                source_format = "html"
                async with aiofiles.open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = await f.read()
                from app.text_cleaning.html_cleaner import HTMLCleaner
                markdown_text = HTMLCleaner(ignore_links=True, ignore_images=False).clean(html)
            else:
                # .txt и прочее текстовое
                source_format = "text"
                async with aiofiles.open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                    markdown_text = await f.read()

        markdown_text = (markdown_text or "").strip()

        return {
            "original_filename": file.filename,
            "file_size": file_size,
            "detected_mime": detected_mime,
            "source_format": source_format,
            "raw_text": markdown_text,
            "raw_text_length": len(markdown_text),
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except (OSError, IOError):
                pass


async def save_file_and_compute_hash(file: UploadFile) -> Tuple[str, int, str]:
    """
    Сохраняет загруженный файл, вычисляет его хеш и валидирует размер и MIME-type.

    Args:
        file (UploadFile): Загруженный файл.

    Returns:
        Tuple[str, int, str]: Путь к временному файлу, размер файла и хеш SHA256.

    Raises:
        HTTPException: Если файл пустой, слишком большой или имеет неподдерживаемый тип.
    """
    logger.info(f"Saving uploaded file: {file.filename}")
    file_ext = os.path.splitext(file.filename)[1]
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    file_size = 0
    hasher = hashlib.sha256()
    max_file_size = settings.max_file_size_mb * 1024 * 1024  # Переводим МБ в байты

    try:
        async with aiofiles.open(temp_path, "wb") as f:
            while chunk := await file.read(8192):
                file_size += len(chunk)
                # Проверяем размер файла во время чтения
                if file_size > max_file_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed size ({settings.max_file_size_mb} MB)"
                    )
                await f.write(chunk)
                hasher.update(chunk)

        if file_size == 0:
            logger.warning(f"Uploaded file is empty: {file.filename}")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Определяем и валидируем MIME-type
        detected_mime = detect_mime_type(temp_path)
        validate_mime_type_and_extension(file.filename, detected_mime)
        logger.info(f"Detected MIME type: {detected_mime}")

        file_hash = hasher.hexdigest()
        logger.info(f"File saved successfully: {temp_path}, size: {file_size} bytes, hash: {file_hash}")
        return temp_path, file_size, file_hash

    except HTTPException:
        # Пробрасываем HTTP исключения без изменений
        raise
    except Exception as e:
        logger.error(f"Failed to save file {file.filename}: {str(e)}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def parse_category_path(category_path: Optional[Union[str, List[str]]]) -> List[str]:
    """
    Парсит путь категории из строки JSON или списка.

    Args:
        category_path (Optional[Union[str, List]]): Путь категории в виде строки JSON или списка.

    Returns:
        List[str]: Список категорий.

    Raises:
        HTTPException: Если формат пути категории недопустим.
    """
    if not category_path:
        logger.debug("No category path provided, returning empty list")
        return []
        
    if isinstance(category_path, list):
        logger.debug(f"Category path is already a list: {category_path}")
        return category_path
        
    try:
        cat_list = json.loads(category_path)
        if not isinstance(cat_list, list):
            error_msg = "category_path must be a JSON array"
            logger.warning(f"Invalid category_path format: {error_msg}")
            raise ValueError(error_msg)
        logger.info(f"Parsed category path from JSON: {cat_list}")
        return cat_list
        
    except Exception as e:
        error_msg = f"Invalid category_path format: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)


def _prepare_source_id_and_categories(
    source_id: Optional[str],
    category_path: Optional[str],
    filename: str
) -> Tuple[str, List[str]]:
    category_list = parse_category_path(category_path) if category_path else []
    effective_source_id = source_id
    if not effective_source_id:
        parts = category_list + [filename]
        effective_source_id = generate_uuid_from_parts(parts)
    return effective_source_id, category_list


async def check_existing_document(collection_name: str, source_id: str, doc_hash: str) -> List[Any]:
    """
    Проверяет существование документа с заданным source_id и doc_hash.

    Args:
        collection_name (str): Название коллекции.
        source_id (str): Идентификатор источника.
        doc_hash (str): Хеш документа.

    Returns:
        List[Any]: Список найденных точек (обычно 0 или 1).
    """
    logger.debug(f"Checking for existing document with source_id={source_id}, doc_hash={doc_hash} in collection {collection_name}")
    
    client = get_client()
    filter_cond = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(key="source_id", match=qdrant_models.MatchValue(value=source_id)),
            qdrant_models.FieldCondition(key="doc_hash", match=qdrant_models.MatchValue(value=doc_hash)),
            qdrant_models.FieldCondition(key="is_latest", match=qdrant_models.MatchValue(value=True)),
        ]
    )
    
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_cond,
            limit=1,
            with_payload=True,
            with_vectors=True
        )
        logger.info(f"Found {len(points)} existing document(s) with matching source_id and doc_hash")
        return points
        
    except UnexpectedResponse as e:
        if "Not found: Collection" in str(e):
            logger.warning(f"Collection {collection_name} not found")
            return []
        logger.error(f"Error checking existing document: {str(e)}")
        raise


async def get_all_points_for_source(collection_name: str, source_id: str) -> List[Any]:
    """
    Получает все точки для заданного source_id.

    Args:
        collection_name (str): Название коллекции.
        source_id (str): Идентификатор источника.

    Returns:
        List[Any]: Список всех точек с заданным source_id.
    """
    logger.debug(f"Getting all points for source_id={source_id} in collection {collection_name}")
    
    client = get_client()
    filter_cond = qdrant_models.Filter(
        must=[qdrant_models.FieldCondition(key="source_id", match=qdrant_models.MatchValue(value=source_id))]
    )
    
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_cond,
            limit=1000,
            with_payload=True,
            with_vectors=True
        )
        logger.info(f"Retrieved {len(points)} points for source_id={source_id}")
        return points
        
    except UnexpectedResponse as e:
        if "Not found: Collection" in str(e):
            logger.warning(f"Collection {collection_name} not found")
            return []
        logger.error(f"Error getting points for source: {str(e)}")
        raise


async def _update_point_payload(
        point: qdrant_models.PointStruct,
        file_path: Optional[str],
        category_list: List[str],
        source_format: str,
        original_filename: str,
        doc_hash: str,
) -> qdrant_models.PointStruct:
    """
    Обновляет payload точки: удаляет старые поля категорий только если переданы новые,
    добавляет общие поля и новые категории (category_levelN, category_id_levelN, category_level).
    """
    logger.debug(f"Updating payload for point {point.id}")

    new_payload = point.payload.copy()

    # Удаляем старые поля категорий ТОЛЬКО если переданы новые категории
    if category_list:
        keys_to_remove = [k for k in new_payload.keys() if k.startswith('category_')]
        for k in keys_to_remove:
            del new_payload[k]
            logger.debug(f"Removed category field: {k}")

    # Добавляем общие поля (не категорийные)
    new_payload.update({
        "file_path": file_path or point.payload.get("file_path"),
        "is_latest": True,
        "source_format": source_format,
        "original_format": source_format,
        "original_filename": original_filename,
        "content_type": source_format,
        "doc_hash": doc_hash,
    })

    # Обработка категорий (только если передан непустой список)
    if category_list:
        # Преобразуем путь категорий в строку
        new_payload["category_path"] = " / ".join(category_list)
        
        # Добавляем уровни категорий (category_level0, category_level1, ...)
        for i, category in enumerate(category_list):
            new_payload[f"category_level{i}"] = category
        # Добавляем поле category_level - номер последнего уровня
        new_payload["category_level"] = len(category_list) - 1
    else:
        # Если категории не переданы, сохраняем существующее значение category_path (если есть)
        if "category_path" not in new_payload:
            new_payload["category_path"] = None
        # Оставляем существующие category_level и category_id_levelN без изменений

    updated_point = qdrant_models.PointStruct(
        id=point.id,
        vector=point.vector,
        payload=new_payload
    )
    logger.debug(f"Successfully updated payload for point {point.id}")
    return updated_point

async def _create_document(
    text_content: str,
    source_id: str,
    version: int,
    source_format: str,
    file: UploadFile,
    file_path: Optional[str],
    temp_path: str,
    category_list: List[str],
    doc_hash: str,
    collection_name: str,
    batch_writer: QdrantBatchWriter,
    mark_old: bool = False,
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Создаёт документ (первую версию или новую) через process_documents.
    Если mark_old=True, предварительно помечает старые версии как неактуальные.
    Возвращает (updated_source_ids, point_ids).

    Args:
        text_content (str): Текстовое содержимое документа.
        source_id (str): Идентификатор источника.
        version (int): Версия документа.
        source_format (str): Формат источника.
        file (UploadFile): Загруженный файл.
        file_path (Optional[str]): Путь к файлу.
        temp_path (str): Временный путь к файлу.
        category_list (List[str]): Список категорий.
        doc_hash (str): Хеш документа.
        collection_name (str): Название коллекции.
        batch_writer (QdrantBatchWriter): Писатель для пакетной записи.
        mark_old (bool): Пометить старые версии как неактуальные.

    Returns:
        Tuple[List[str], List[str]]: Кортеж из списков обновленных source_id и идентификаторов точек.
    """
    logger.info(f"Creating new document version {version} for source_id={source_id}")

    if mark_old:
        logger.debug(f"Marking old versions as not latest for source_id={source_id}, keeping version {version}")
        batch_writer.mark_old_versions_not_latest(collection_name, source_id, keep_version=version)

    doc_metadata = {
        "source_id": source_id,
        "source_format": source_format,
        "original_format": source_format,
        "version": version,
        "is_latest": True,
        "original_filename": file.filename,
        "file_path": file_path or temp_path,
        "content_type": source_format,
        "doc_hash": doc_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # category_path передается отдельно в DocumentCreate, не добавляем в payload

    doc_request = DocumentsUploadRequest(
        documents=[DocumentCreate(
            text=text_content,
            payload=doc_metadata,
            version=version,
            category_path=category_list if category_list else None,
            filename=file.filename,
            title=None
        )],
        collection_name=collection_name
    )
    
    try:
        updated_source_ids, point_ids, skipped_docs = await process_documents(doc_request, batch_writer)
        logger.info(f"Successfully created document version {version}, got {len(point_ids)} point(s)")
        return updated_source_ids, point_ids, skipped_docs
    except Exception as e:
        logger.error(f"Failed to create document version {version} for source_id={source_id}: {str(e)}")
        raise


async def process_single_file(
    file: UploadFile,
    collection_name: str,
    source_format: str,
    batch_writer: QdrantBatchWriter,
    source_id: Optional[str] = None,
    file_path: Optional[str] = None,
    category_path: Optional[str] = None,
    skip_if_exists: bool = False,
) -> dict:
    """
    Обрабатывает один загруженный файл: сохраняет, проверяет дубликаты,
    создает или обновляет документ в коллекции.

    Args:
        file (UploadFile): Загруженный файл.
        collection_name (str): Название коллекции для сохранения.
        source_format (str): Формат источника (например, 'text', 'pdf').
        batch_writer (QdrantBatchWriter): Писатель для пакетной записи в Qdrant.
        source_id (Optional[str]): Идентификатор источника. Если не указан, генерируется.
        file_path (Optional[str]): Путь к файлу. Если не указан, используется временный путь.
        category_path (Optional[str]): Путь категорий в виде JSON строки или списка.
        skip_if_exists (bool): Пропускать ли файл, если он уже существует.

    Returns:
        dict: Результат обработки с информацией о статусе, source_id, версии и других деталях.
    """
    start_time = time.time()
    temp_path = None
    result = {
        "status": "error",
        "source_id": None,
        "version": None,
        "is_latest": None,
        "content_hash": None,
        "uploaded_chunks": 0,
        "chunk_ids": [],
        "error_message": None
    }

    try:
        logger.info(f"Start processing file {file.filename}")
        temp_path, file_size, doc_hash = await save_file_and_compute_hash(file)
        # Убрано логирование сохранения файла - операция быстрая и надежная

        effective_source_id, category_list = _prepare_source_id_and_categories(source_id, category_path, file.filename)

        existing_points = await check_existing_document(collection_name, effective_source_id, doc_hash)
        # Убрано логирование количества существующих точек - операция быстрая и не критичная

        # Сценарий A: точное совпадение
        if existing_points:
            if skip_if_exists:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_path}: {e}")
                result.update({
                    "status": "skipped",
                    "source_id": effective_source_id,
                    "version": existing_points[0].payload.get("version"),
                    "is_latest": True,
                    "content_hash": doc_hash
                })
                return result
            else:
                batch_writer.mark_old_versions_not_latest(
                    collection_name, effective_source_id,
                    keep_version=existing_points[0].payload.get("version")
                )
                for point in existing_points:
                    updated_point = await _update_point_payload(
                        point, file_path, category_list, source_format,
                        file.filename, doc_hash
                    )
                    batch_writer.add_point(collection_name, updated_point)
                os.unlink(temp_path)
                # Обновление результата (повторяющийся фрагмент с другими сценариями)
                result.update({
                    "status": "updated",
                    "uploaded_chunks": len(existing_points),
                    "chunk_ids": [p.id for p in existing_points],
                    "source_id": effective_source_id,
                    "version": existing_points[0].payload.get("version"),
                    "is_latest": True,
                    "content_hash": doc_hash
                })
                return result

        # Сценарий B и C
        text_content = await extract_text_from_file(temp_path)
        if not text_content:
            result.update({
                "status": "file is null, skipped",
                "source_id": None,
                "version": None,
                "is_latest": None,
                "content_hash": None,
                "uploaded_chunks": 0,
                "chunk_ids": [],
                "error_message": None
            })
            return result

        all_points = await get_all_points_for_source(collection_name, effective_source_id)

        if all_points:
            max_version = max(p.payload.get("version", 1) for p in all_points)
            current_point = next(p for p in all_points if p.payload.get("version") == max_version)
            existing_hash = current_point.payload.get("doc_hash")

            if existing_hash == doc_hash:
                # Убрано логирование совпадения содержимого - операция быстрая и не требует мониторинга
                batch_writer.mark_old_versions_not_latest(
                    collection_name, effective_source_id, keep_version=max_version
                )
                updated_point = await _update_point_payload(
                    current_point, file_path, category_list, source_format,
                    file.filename, doc_hash
                )
                batch_writer.add_point(collection_name, updated_point)
                os.unlink(temp_path)
                # Обновление результата (повторяющийся фрагмент с другими сценариями)
                result.update({
                    "status": "updated",
                    "uploaded_chunks": 1,
                    "chunk_ids": [current_point.id],
                    "source_id": effective_source_id,
                    "version": max_version,
                    "is_latest": True,
                    "content_hash": doc_hash
                })
                return result
            else:
                # Создание новой версии
                new_version = max_version + 1
                start_create_time = time.time()
                # Убрано логирование начала создания новой версии - оставлено только время выполнения
                updated_source_ids, point_ids, skipped_docs = await _create_document(
                    text_content, effective_source_id, new_version, source_format,
                    file, file_path, temp_path, category_list, doc_hash,
                    collection_name, batch_writer, mark_old=True
                )
                create_elapsed = time.time() - start_create_time
                logger.info(f"Document version {new_version} with {len(point_ids)} chunks created in {create_elapsed:.3f}s")
                os.unlink(temp_path)
                result.update({
                    "status": "created",
                    "uploaded_chunks": len(point_ids),
                    "chunk_ids": point_ids,
                    "source_id": effective_source_id,
                    "version": new_version,
                    "is_latest": True,
                    "content_hash": doc_hash
                })
                return result
        else:
            # Первая версия
            new_version = 1
            start_create_time = time.time()
            # Убрано логирование начала создания первой версии - оставлено только время выполнения
            updated_source_ids, point_ids, skipped_docs = await _create_document(
                text_content, effective_source_id, new_version, source_format,
                file, file_path, temp_path, category_list, doc_hash,
                collection_name, batch_writer, mark_old=False
            )
            create_elapsed = time.time() - start_create_time
            logger.info(f"Document version {new_version} with {len(point_ids)} chunks created in {create_elapsed:.3f}s")
            os.unlink(temp_path)
            result.update({
                "status": "created",
                "uploaded_chunks": len(point_ids),
                "chunk_ids": point_ids,
                "source_id": effective_source_id,
                "version": new_version,
                "is_latest": True,
                "content_hash": doc_hash
            })
            return result

    except Exception as e:
        if isinstance(e, HTTPException):
            error_msg = f"HTTP {e.status_code}: {e.detail}"
            logger.error(error_msg)
        elif isinstance(e, ValueError):
            error_msg = f"Invalid value: {str(e)}"
            logger.error(error_msg)
        elif isinstance(e, IOError):
            error_msg = f"File I/O error: {str(e)}"
            logger.error(error_msg)
        else:
            elapsed = time.time() - start_time
            error_msg = f"Upload failed after {elapsed:.3f}s: {str(e)}"
            logger.error(error_msg, exc_info=True)
        result["error_message"] = error_msg
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except (OSError, IOError):
                pass

    # Prepare standardized response using response utility
    response = create_file_upload_response(
        status=result.get("status"),
        source_id=result.get("source_id"),
        version=result.get("version"),
        is_latest=bool(result.get("is_latest")),
        content_hash=result.get("content_hash"),
        uploaded_chunks=result.get("uploaded_chunks", 0),
        chunk_ids=result.get("chunk_ids", []),
        original_filename=file.filename,
        collection_name=collection_name,
        error_code="processing_error" if result.get("error_message") else None,
        error_message=result.get("error_message")
    )

    # Добавляем поле status в data, если оно существует
    data = response.get("data")
    if data is not None and "status" not in data:
        data["status"] = result.get("status")

    return response

@router.post("/upload/batch",
    summary="Пакетная загрузка файлов",
    description=load_openapi_md("files_upload_batch.md"),
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files", "collection_name", "metadata"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Выберите один или несколько файлов"
                            },
                            "collection_name": {
                                "type": "string",
                                "description": "Имя коллекции в Qdrant",
                                "example": "documents"
                            },
                            "skip_if_exists": {
                                "type": "boolean",
                                "default": False,
                                "description": "Пропустить файл, если он уже существует (по хешу)"
                            },
                            "metadata": {
                                "type": "string",
                                "description": "JSON-массив метаданных для каждого файла",
                                "example": "[{\"source_format\":\"pdf\"},{\"source_format\":\"text\"}]"
                            }
                        }
                    }
                }
            }
        }
    })
async def upload_files_batch(
    files: List[UploadFile] = File(...),
    collection_name: str = Form(...),
    skip_if_exists: bool = Form(False),
    metadata: str = Form(...),
):
    try:
        metadatas = json.loads(metadata)
        if not isinstance(metadatas, list) or len(metadatas) != len(files):
            raise ValueError("metadata must be a JSON array with the same length as files")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {e}")

    batch_writer = QdrantBatchWriter()
    results = []
    for i, (file, meta) in enumerate(zip(files, metadatas)):
        source_id = meta.get("source_id")
        file_path = meta.get("file_path")
        source_format = meta.get("source_format")
        category_path = meta.get("category_path")

        result = await process_single_file(
            file=file,
            collection_name=collection_name,
            source_format=source_format,
            batch_writer=batch_writer,
            source_id=source_id,
            file_path=file_path,
            category_path=category_path,
            skip_if_exists=skip_if_exists,
        )
        results.append(result)

    await batch_writer.commit()
    # Prepare standardized response for batch upload using response utility
    batch_results = []
    for result in results:
        batch_result = create_file_upload_response(
            status=result.get("status"),
            source_id=result.get("source_id"),
            version=result.get("version"),
            is_latest=result.get("is_latest"),
            content_hash=result.get("content_hash"),
            uploaded_chunks=result.get("uploaded_chunks", 0),
            chunk_ids=result.get("chunk_ids", []),
            original_filename=result.get("original_filename"),
            collection_name=collection_name,
            error_code="processing_error" if result.get("error_message") else None,
            error_message=result.get("error_message")
        )
        batch_results.append(batch_result)
    
    return create_batch_upload_response(batch_results, len(files))


@router.post("/upload", 
    summary="Загрузка одного файла",
    description=load_openapi_md("files_upload_single.md"))
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    source_format: str = Form(...),
    source_id: Optional[str] = Form(None),
    file_path: Optional[str] = Form(None),
    category_path: Optional[str] = Form(None),
    skip_if_exists: bool = Form(False),
):
    batch_writer = QdrantBatchWriter()
    result = await process_single_file(
        file=file,
        collection_name=collection_name,
        source_format=source_format,
        batch_writer=batch_writer,
        source_id=source_id,
        file_path=file_path,
        category_path=category_path,
        skip_if_exists=skip_if_exists,
    )
    await batch_writer.commit()
    return result


@router.post(
    "/preview/md",
    summary="Проверка конвертации файла в Markdown",
    description="Принимает файл и возвращает raw_text (Markdown) до разбиения на чанки и без загрузки в Qdrant.",
)
async def preview_file_markdown(
    file: UploadFile = File(...),
):
    try:
        data = await convert_file_to_markdown_raw(file)
        return create_response(success=True, data=data)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Markdown preview failed")
        return create_response(success=False, error_code="markdown_preview_failed", error_message=str(e))


@router.post(
    "/preview/md/text",
    summary="Проверка конвертации файла в Markdown (plain text)",
    description="Возвращает raw_text как text/markdown, чтобы переносы строк отображались без экранирования JSON.",
    response_class=PlainTextResponse,
)
async def preview_file_markdown_text(
    file: UploadFile = File(...),
):
    data = await convert_file_to_markdown_raw(file)
    # PlainTextResponse покажет переносы строк как есть; content-type будет text/plain,
    # но для клиентов это достаточно для корректного отображения.
    return data.get("raw_text", "")