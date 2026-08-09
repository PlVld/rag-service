"""
Кэшированный синглтон DoclingCleaner.

Предотвращает повторную инициализацию DocumentConverter на каждый запрос.
Один экземпляр создаётся на старте приложения с параметрами из settings.
"""

import logging
from typing import Optional

from app.core.config import settings

from .base import BaseCleaner

logger = logging.getLogger(__name__)

# Global cached instance
_docling_cleaner: Optional["BaseCleaner"] = None


def _create_cleaner() -> "BaseCleaner":
    """Creates a new DoclingCleaner instance with settings parameters."""
    from app.text_cleaning.docling_cleaner import DoclingCleaner

    cleaner = DoclingCleaner(
        do_ocr=settings.docling_do_ocr,
        ocr_engine=settings.docling_ocr_engine,
        image_description_model=settings.docling_image_description_model or None,
        images_scale=settings.docling_images_scale,
    )
    logger.info(
        "DoclingCleaner singleton created "
        f"(ocr={settings.docling_do_ocr}, engine={settings.docling_ocr_engine}, "
        f"scale={settings.docling_images_scale})"
    )
    return cleaner


def get_docling_cleaner() -> "BaseCleaner":
    """
    Returns a cached DoclingCleaner instance.

    The instance is created lazily on the first call and reused on subsequent calls.
    """
    global _docling_cleaner
    if _docling_cleaner is None:
        _docling_cleaner = _create_cleaner()
    return _docling_cleaner


def reset_docling_cleaner() -> None:
    """
    Resets the cached DoclingCleaner instance.

    Useful for testing or if settings change at runtime.
    """
    global _docling_cleaner
    if _docling_cleaner is not None:
        logger.info("Resetting DoclingCleaner singleton")
        _docling_cleaner = None
