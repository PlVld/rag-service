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
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
        HTMLFormatOption,
        ImageFormatOption,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractOcrOptions,
        TesseractCliOcrOptions,
        EasyOcrOptions,
        RapidOcrOptions,
        PictureDescriptionApiOptions,
    )
    from docling.datamodel.backend_options import HTMLBackendOptions
    from docling.utils.api_image_request import api_image_request
    from docling_core.types.doc import (
        ImageRefMode,
        PictureMeta,
        DescriptionMetaField,
    )
    from docling.datamodel.settings import settings as docling_settings

    DOCLING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Docling not available: {e}")
    DOCLING_AVAILABLE = False
    # Определяем типы как None для type checking
    DocumentConverter = None  # type: ignore
    PdfFormatOption = None  # type: ignore
    HTMLFormatOption = None  # type: ignore
    ImageFormatOption = None  # type: ignore
    InputFormat = None  # type: ignore
    PdfPipelineOptions = None  # type: ignore
    TesseractOcrOptions = None  # type: ignore
    TesseractCliOcrOptions = None  # type: ignore
    EasyOcrOptions = None  # type: ignore
    RapidOcrOptions = None  # type: ignore
    PictureDescriptionApiOptions = None  # type: ignore
    HTMLBackendOptions = None  # type: ignore
    api_image_request = None  # type: ignore
    ImageRefMode = None  # type: ignore
    PictureMeta = None  # type: ignore
    DescriptionMetaField = None  # type: ignore
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
        image_description_url: str = "",
        image_description_api_key: str = "",
        image_description_prompt: str = "",
        image_description_timeout: float = 60.0,
        image_description_extra_params: Optional[dict] = None,
        images_scale: float = 1.0,
        extract_pdf_images: bool = False,
    ):
        self.do_ocr = do_ocr
        self.ocr_engine = ocr_engine
        self.image_description_model = image_description_model
        self.image_description_url = image_description_url
        self.image_description_api_key = image_description_api_key
        self.image_description_prompt = image_description_prompt
        self.image_description_timeout = image_description_timeout
        self.image_description_extra_params = image_description_extra_params or {}
        self.images_scale = images_scale
        self.extract_pdf_images = extract_pdf_images
        # None — проба ещё не выполнялась; результат кэшируется, чтобы неработающий
        # сервер не проверялся заново на каждую конвертацию
        self._image_description_available: Optional[bool] = None
        self._converter: Any = None

    def probe_image_description_api(self, timeout: float = 10.0) -> bool:
        """
        Проверяет доступность OpenAI-совместимого сервера описания изображений.

        Вызывается при старте сервиса; если сервер недоступен, конвертация
        продолжается без описаний картинок.
        """
        if not self.image_description_model or not self.image_description_url:
            self._image_description_available = False
            return False

        base_url = self.image_description_url
        for suffix in ("/chat/completions", "/completions"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        models_url = base_url.rstrip("/") + "/models"
        headers = (
            {"Authorization": f"Bearer {self.image_description_api_key}"}
            if self.image_description_api_key
            else {}
        )
        try:
            import httpx

            response = httpx.get(models_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            self._image_description_available = True
            logger.info(
                f"Image description API available at {self.image_description_url} "
                f"(model: {self.image_description_model})"
            )
        except Exception as err:
            self._image_description_available = False
            logger.warning(
                f"Image description API unavailable ({models_url}): {err}. "
                "Images will be converted without descriptions."
            )
        return self._image_description_available

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

        # Настройка описания изображений через OpenAI-совместимый API.
        # Для описания Docling должен отрендерить картинки, поэтому generate_picture_images
        # включается принудительно, даже если extract_pdf_images=False.
        if self.image_description_model and self._image_description_available is None:
            self.probe_image_description_api()
        if self.image_description_model and self._image_description_available:
            headers = (
                {"Authorization": f"Bearer {self.image_description_api_key}"}
                if self.image_description_api_key
                else {}
            )
            # generate_picture_images: VLM нужен растр картинки;
            # do_picture_description включает сам этап описания;
            # enable_remote_services: без него Docling запрещает внешние API-запросы
            pdf_pipeline_options.generate_picture_images = True
            pdf_pipeline_options.do_picture_description = True
            pdf_pipeline_options.enable_remote_services = True
            pdf_pipeline_options.picture_description_options = PictureDescriptionApiOptions(
                url=self.image_description_url,
                headers=headers,
                params={"model": self.image_description_model, **self.image_description_extra_params},
                prompt=self.image_description_prompt or "Describe this image in a few sentences.",
                timeout=self.image_description_timeout,
            )
            logger.info(
                f"Picture description enabled via API model: {self.image_description_model}"
            )

        # Создание конвертера с новым API (format_options вместо pipeline_options)
        # IMAGE (отдельные PNG/JPEG) идёт через тот же StandardPdfPipeline —
        # передаём те же опции, чтобы картинки описывались и там
        image_format_option = ImageFormatOption(pipeline_options=pdf_pipeline_options)
        # SimplePipeline-форматы (DOCX, HTML) не имеют стадии описания в Docling —
        # описания добавляются пост-обработкой в _describe_pictures.
        # Для HTML нужен fetch_images, иначе вместо картинок будут плейсхолдеры.
        html_backend_options = None
        if self.image_description_model and self._image_description_available:
            html_backend_options = HTMLBackendOptions(fetch_images=True)
        html_format_option = (
            HTMLFormatOption(backend_options=html_backend_options)
            if html_backend_options is not None
            else HTMLFormatOption()
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
                InputFormat.IMAGE: image_format_option,
                InputFormat.HTML: html_format_option,
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

    def _describe_pictures(self, document: Any) -> None:
        """
        Добавляет описания картинок для форматов без enrichment-стадии (DOCX, HTML).

        Docling запускает этап do_picture_description только в PDF-пайплайне;
        для SimplePipeline-форматов вызываем API сами и пишем результат в
        meta.description — оттуда описание попадает в Markdown.
        """
        headers = (
            {"Authorization": f"Bearer {self.image_description_api_key}"}
            if self.image_description_api_key
            else {}
        )
        prompt = self.image_description_prompt or "Describe this image in a few sentences."
        described = 0
        for item in document.pictures:
            meta = item.meta
            if meta is not None and meta.description and meta.description.text:
                continue
            ref = item.image
            if ref is None:
                continue
            try:
                pil_image = ref.pil_image
            except Exception as err:
                logger.warning(f"Could not decode picture: {err}")
                continue
            if pil_image is None:
                continue
            try:
                result = api_image_request(
                    image=pil_image.convert("RGB"),
                    prompt=prompt,
                    url=self.image_description_url,
                    timeout=self.image_description_timeout,
                    headers=headers,
                    model=self.image_description_model,
                    **self.image_description_extra_params,
                )
            except Exception as err:
                logger.warning(f"Picture description API call failed: {err}")
                continue
            if not result.text:
                logger.warning("Picture description API returned empty text")
                continue
            if item.meta is None:
                item.meta = PictureMeta()
            item.meta.description = DescriptionMetaField(
                text=result.text, created_by="api"
            )
            described += 1
        if described:
            logger.info(f"Described {described} picture(s) via API post-processing")

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

            # Описания для форматов без enrichment-стадии (DOCX, HTML)
            if (
                self.image_description_model
                and self._image_description_available
                and result.input.format != InputFormat.PDF
                and result.input.format != InputFormat.IMAGE
            ):
                self._describe_pictures(result.document)

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
