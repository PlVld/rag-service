import gc
import logging
import shutil
import sys
from io import BytesIO
from typing import Optional, Union, Any
from pathlib import Path

from .base import BaseCleaner
from .bundled_tools import bundled_tessdata_dir, bundled_tesseract_cmd
from .image_store import IMAGES_SUBDIR

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractOcrOptions,
        TesseractCliOcrOptions,
        EasyOcrOptions,
        RapidOcrOptions,
    )
    from docling_core.types.doc import ImageRefMode
    from docling.datamodel.settings import settings as docling_settings

    DOCLING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Docling not available: {e}")
    DOCLING_AVAILABLE = False
    # Определяем типы как None для type checking
    DocumentConverter = None  # type: ignore
    PdfFormatOption = None  # type: ignore
    InputFormat = None  # type: ignore
    PdfPipelineOptions = None  # type: ignore
    TesseractOcrOptions = None  # type: ignore
    TesseractCliOcrOptions = None  # type: ignore
    EasyOcrOptions = None  # type: ignore
    RapidOcrOptions = None  # type: ignore
    ImageRefMode = None  # type: ignore
    docling_settings = None  # type: ignore

logger = logging.getLogger(__name__)

# Модели, скачанные через `docling-tools models download -o ./models`. Путь считается
# от корня проекта, поэтому переносится вместе с ним на другую машину.
LOCAL_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _local_artifacts_path() -> Optional[Path]:
    """
    Возвращает каталог с локальными моделями, если он есть.

    Если artifacts_path задан, docling больше не обращается в сеть и падает с
    FileNotFoundError на отсутствующей модели, поэтому при пустом каталоге отдаём None
    и оставляем штатную загрузку с HuggingFace.
    """
    if not LOCAL_MODELS_DIR.is_dir():
        return None
    if not any(LOCAL_MODELS_DIR.iterdir()):
        return None
    return LOCAL_MODELS_DIR


def _disable_torch_compile_without_c_compiler() -> None:
    """
    Docling по умолчанию оборачивает layout-модель в torch.compile, а backend inductor
    на Windows требует MSVC cl.exe. Без него конвертация падает целиком, поэтому
    заранее переводим модели в eager-режим.
    """
    if docling_settings is None:
        return
    if not docling_settings.inference.compile_torch_models:
        return

    compiler = "cl" if sys.platform == "win32" else "g++"
    if shutil.which(compiler) is None:
        docling_settings.inference.compile_torch_models = False
        logger.info(f"C++ compiler '{compiler}' not found, running Docling models in eager mode")


class DoclingCleaner(BaseCleaner):
    """
    Конвертирует документы (PDF, DOCX, HTML, изображения) в Markdown с помощью Docling.
    Распознаёт структуру документа, таблицы, изображения и описывает их содержимое.
    """

    def __init__(
        self,
        do_ocr: bool = True,
        ocr_engine: str = "easyocr",
        image_description_model: Optional[str] = None,
        images_scale: float = 1.0,
        extract_pdf_images: bool = False,
    ):
        self.do_ocr = do_ocr
        self.ocr_engine = ocr_engine
        self.image_description_model = image_description_model
        self.images_scale = images_scale
        self.extract_pdf_images = extract_pdf_images
        self._converter: Any = None

    def _get_converter(self) -> Optional[DocumentConverter]:
        if self._converter is not None:
            return self._converter

        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is not installed. Run: pip install docling")

        _disable_torch_compile_without_c_compiler()

        # Настройка PDF pipeline
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = self.do_ocr
        pdf_pipeline_options.images_scale = self.images_scale
        # Рендеринг страниц ради картинок дорог по CPU и памяти, поэтому под флагом
        pdf_pipeline_options.generate_picture_images = self.extract_pdf_images

        artifacts_path = _local_artifacts_path()
        if artifacts_path is not None:
            pdf_pipeline_options.artifacts_path = artifacts_path
            logger.info(f"Using local Docling models from: {artifacts_path}")

        # Настройка OCR
        if self.do_ocr:
            if self.ocr_engine == "tesseract":
                # TesseractCliOcrOptions вызывает tesseract как CLI (не требует tesserocr)
                tesseract_kwargs: dict = {}
                bundled_cmd = bundled_tesseract_cmd()
                if bundled_cmd is not None:
                    tesseract_kwargs["tesseract_cmd"] = bundled_cmd
                    tesseract_kwargs["path"] = bundled_tessdata_dir()
                    logger.info(f"Using bundled Tesseract: {bundled_cmd}")
                ocr_options = TesseractCliOcrOptions(lang=["rus", "eng"], **tesseract_kwargs)
                pdf_pipeline_options.ocr_options = ocr_options
                logger.info("Using Tesseract CLI OCR engine (rus, eng)")
            elif self.ocr_engine == "rapidocr":
                ocr_options = RapidOcrOptions(lang=["ru", "en"])
                pdf_pipeline_options.ocr_options = ocr_options
                logger.info("Using RapidOCR engine (ru, en)")
            else:
                ocr_options = EasyOcrOptions()
                pdf_pipeline_options.ocr_options = ocr_options
                logger.info("Using EasyOCR engine")

        # Создание конвертера с новым API (format_options вместо pipeline_options)
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            },
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.IMAGE,
            ],
        )

        logger.info("Docling DocumentConverter initialized")
        return self._converter

    def clean(self, source: Union[str, bytes, Path], media_dir: Optional[Path] = None, **kwargs) -> str:
        """
        Конвертирует документ в Markdown.

        Args:
            source: Путь к файлу (str/Path) или байтовое содержимое (bytes)
            media_dir: Каталог для извлечённых изображений. Если задан, картинки
                сохраняются в {media_dir}/images, а в Markdown появляются
                относительные ссылки на них.

        Returns:
            Markdown-представление документа
        """
        if not source:
            return ""

        try:
            converter = self._get_converter()
            logger.info(f"Starting Docling conversion, input type: {type(source)}")

            # Обработка разных типов входных данных
            if isinstance(source, bytes):
                # Для байтовых данных создаём временный файл или используем BytesIO
                source_obj = BytesIO(source)
            elif isinstance(source, (str, Path)):
                source_obj = Path(source) if isinstance(source, str) else source
                if not source_obj.exists():
                    logger.error(f"File not found: {source}")
                    return ""
            else:
                logger.error(f"Unsupported source type: {type(source)}")
                return ""

            # Конвертация (converter гарантированно не None, так как проверено DOCLING_AVAILABLE)
            result = converter.convert(source_obj)  # type: ignore

            if result is None:
                logger.warning("Docling returned None result")
                return ""

            # Извлечение Markdown
            if media_dir is not None:
                markdown_content = self._export_with_images(result.document, media_dir)
            else:
                markdown_content = result.document.export_to_markdown()

            if not markdown_content:
                logger.warning("Docling produced empty markdown")
                return ""

            logger.info(f"Docling conversion successful, output length: {len(markdown_content)}")
            return markdown_content.strip()

        except Exception as err:
            logger.error(f"Docling conversion failed: {err}", exc_info=True)
            raise
        finally:
            # Освобождаем память после конвертации
            self._clear_converter()
            gc.collect()

    @staticmethod
    def _export_with_images(document: Any, media_dir: Path) -> str:
        """
        Сохраняет Markdown вместе с картинками и возвращает его текст.

        export_to_markdown(image_mode=REFERENCED) файлы на диск не пишет: сериализатор
        откатывается на плейсхолдер, если uri — это data:-URI. Картинки сохраняет только
        save_as_markdown, поэтому пишем во временный document.md и читаем его обратно.
        Относительный artifacts_dir даёт ссылки вида images/image_000000_<hash>.png.
        """
        md_path = media_dir / "document.md"
        try:
            document.save_as_markdown(
                md_path,
                artifacts_dir=Path(IMAGES_SUBDIR),
                image_mode=ImageRefMode.REFERENCED,
            )
            return md_path.read_text(encoding="utf-8")
        finally:
            # Каталог раздаётся публично, полный текст документа там оставлять нельзя
            md_path.unlink(missing_ok=True)

    def _clear_converter(self):
        """Освобождает память конвертера для предотвращения утечек памяти."""
        if self._converter is not None:
            try:
                del self._converter
            except (AttributeError, NameError):
                pass
            self._converter = None
