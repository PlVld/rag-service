"""
Хранилище изображений, извлечённых из документов.

Раскладка на диске:
    {media_dir}/{source_id}/{doc_hash[:16]}/images/image_000000_<hexhash>.png

Каталог адресуется хешем содержимого документа, поэтому повторная загрузка того же
файла попадает в тот же каталог, а новая версия — в новый.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

IMAGES_SUBDIR = "images"
PREVIEW_SOURCE_ID = "_preview"

# Ссылки, которые docling выдаёт при относительном artifacts_dir: ![Image](images/xxx.png).
# На Windows разделителем оказывается обратный слеш, поэтому принимаем оба.
_RELATIVE_IMAGE_LINK = re.compile(r"(!\[[^\]]*]\()(" + IMAGES_SUBDIR + r"[/\\][^)]+)(\))")


def _hash_segment(doc_hash: str) -> str:
    return doc_hash[:16]


def media_root() -> Path:
    return Path(settings.media_dir)


def resolve_media_dir(source_id: str, doc_hash: str) -> Path:
    """Создаёт и возвращает каталог для изображений конкретной версии документа."""
    path = media_root() / source_id / _hash_segment(doc_hash)
    path.mkdir(parents=True, exist_ok=True)
    return path


def public_url_for(source_id: str, doc_hash: str, rel_path: str) -> str:
    """Собирает публичный URL картинки. Разделитель всегда '/', в том числе на Windows."""
    prefix = settings.media_url_prefix.rstrip("/")
    base = settings.media_base_url.rstrip("/")
    rel = rel_path.replace("\\", "/").lstrip("/")
    return f"{base}{prefix}/{source_id}/{_hash_segment(doc_hash)}/{rel}"


def rewrite_image_links(markdown: str, source_id: str, doc_hash: str) -> str:
    """Заменяет относительные ссылки на изображения публичными URL."""
    if not markdown:
        return markdown

    def replace(match: re.Match) -> str:
        return f"{match.group(1)}{public_url_for(source_id, doc_hash, match.group(2))}{match.group(3)}"

    return _RELATIVE_IMAGE_LINK.sub(replace, markdown)


def cleanup_other_versions(source_id: str, keep_doc_hash: str) -> None:
    """
    Удаляет каталоги изображений для остальных версий документа.

    Чанки старых версий помечены is_latest=False и поиском не возвращаются,
    поэтому их картинки недостижимы.
    """
    source_dir = media_root() / source_id
    if not source_dir.is_dir():
        return

    keep = _hash_segment(keep_doc_hash)
    for child in source_dir.iterdir():
        if not child.is_dir() or child.name == keep:
            continue
        try:
            shutil.rmtree(child)
            logger.info(f"Removed stale media directory: {child}")
        except OSError as e:
            logger.warning(f"Failed to remove stale media directory {child}: {e}")


def prepare_media_dir(source_id: Optional[str], doc_hash: Optional[str]) -> Optional[Path]:
    """
    Возвращает каталог для изображений или None, если извлечение отключено
    либо документ не идентифицирован.
    """
    if not settings.docling_extract_images or not source_id or not doc_hash:
        return None
    return resolve_media_dir(source_id, doc_hash)
