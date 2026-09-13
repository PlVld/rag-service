"""
Кэшированный синглтон DoclingCleaner.

Предотвращает повторную инициализацию DocumentConverter на каждый запрос.
Один экземпляр создаётся на старте приложения с параметрами из settings.
"""

import json
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

    image_description_host = settings.docling_image_description_host.rstrip("/")
    image_description_url = (
        f"{image_description_host}:{settings.docling_image_description_port}"
        "/v1/chat/completions"
    )

    # Доп. параметры запроса (JSON): {"chat_template_kwargs": {"enable_thinking": false}}
    extra_params: dict = {}
    raw_params = settings.docling_image_description_extra_params.strip()
    if raw_params:
        try:
            parsed = json.loads(raw_params)
            if isinstance(parsed, dict):
                extra_params = parsed
            else:
                logger.warning(
                    "DOCLING_IMAGE_DESCRIPTION_EXTRA_PARAMS must be a JSON object, got %s; ignored",
                    type(parsed).__name__,
                )
        except json.JSONDecodeError as err:
            logger.warning(
                f"Invalid JSON in DOCLING_IMAGE_DESCRIPTION_EXTRA_PARAMS, ignored: {err}"
            )

    cleaner = DoclingCleaner(
        do_ocr=settings.docling_do_ocr,
        ocr_engine=settings.docling_ocr_engine,
        image_description_model=settings.docling_image_description_model or None,
        image_description_url=image_description_url,
        image_description_api_key=settings.docling_image_description_api_key,
        image_description_prompt=settings.docling_image_description_prompt,
        image_description_timeout=settings.docling_image_description_timeout,
        image_description_extra_params=extra_params,
        images_scale=settings.docling_images_scale,
        extract_pdf_images=settings.docling_extract_pdf_images,
    )
    logger.info(
        "DoclingCleaner singleton created "
        f"(ocr={settings.docling_do_ocr}, engine={settings.docling_ocr_engine}, "
        f"scale={settings.docling_images_scale}, "
        f"pdf_images={settings.docling_extract_pdf_images})"
    )
    return cleaner


def probe_image_description_api() -> bool:
    """Checks the image description API on the cached cleaner (no-op for other cleaners)."""
    from app.text_cleaning.docling_cleaner import DoclingCleaner

    cleaner = get_docling_cleaner()
    if isinstance(cleaner, DoclingCleaner):
        return cleaner.probe_image_description_api()
    return False


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
