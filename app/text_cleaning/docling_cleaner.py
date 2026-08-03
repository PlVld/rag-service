import gc
import logging
from io import BytesIO
from typing import Optional, Union, Any
from pathlib import Path

from .base import BaseCleaner

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

logger = logging.getLogger(__name__)


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
    ):
        self.do_ocr = do_ocr
        self.ocr_engine = ocr_engine
        self.image_description_model = image_description_model
        self.images_scale = images_scale
        self._converter: Any = None

    def _get_converter(self) -> Optional[DocumentConverter]:
        if self._converter is not None:
            return self._converter

        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is not installed. Run: pip install docling")

        # Настройка PDF pipeline
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = self.do_ocr
        pdf_pipeline_options.images_scale = self.images_scale

        # Настройка OCR
        if self.do_ocr:
            if self.ocr_engine == "tesseract":
                # TesseractCliOcrOptions вызывает tesseract как CLI (не требует tesserocr)
                ocr_options = TesseractCliOcrOptions(lang=["rus", "eng"])
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

    def clean(self, source: Union[str, bytes, Path], **kwargs) -> str:
        """
        Конвертирует документ в Markdown.

        Args:
            source: Путь к файлу (str/Path) или байтовое содержимое (bytes)

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

    def _clear_converter(self):
        """Освобождает память конвертера для предотвращения утечек памяти."""
        if self._converter is not None:
            try:
                del self._converter
            except (AttributeError, NameError):
                pass
            self._converter = None
