import logging
import os
import binascii
from io import BytesIO

from pypdf import PdfReader
import pdfplumber  # новая зависимость
from .base import BaseCleaner

try:
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_bytes, convert_from_path

    OCR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OCR libraries not available: {e}")
    pytesseract = None
    convert_from_bytes = None
    convert_from_path = None
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)


class PDFCleaner(BaseCleaner):
    """
    Очищает PDF-документ, извлекая текст и таблицы, преобразуя в Markdown.
    Поддерживает текстовые и отсканированные PDF (OCR).
    """

    def __init__(self, ignore_images: bool = True, ocr_enabled: bool = True, ocr_language: str = 'rus+eng'):
        self.ignore_images = ignore_images
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.poppler_path = None
        self._pdf_bytes = None  # для OCR и pdfplumber

        if OCR_AVAILABLE and ocr_enabled:
            try:
                # Кроссплатформенное определение пути к poppler
                self.poppler_path = None
                import platform
                if platform.system() == "Windows":
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    poppler_path = os.path.join(base_dir, "poppler", "Library", "bin")
                else:
                    # Linux: использует системный poppler или переменную окружения
                    poppler_path = os.environ.get("POPPLER_PATH", "/usr/bin")

                poppler_path = os.path.normpath(poppler_path)
                if os.path.exists(poppler_path):
                    self.poppler_path = poppler_path
                    logger.info(f"Poppler path set to: {self.poppler_path}")
                else:
                    logger.error(f"Poppler path does not exist: {poppler_path}")
                    self.ocr_enabled = False
                if pytesseract:
                    version = pytesseract.get_tesseract_version()
                    logger.info(f"Tesseract version: {version}")
            except Exception as init_e:
                logger.warning(f"OCR initialization failed: {init_e}")
                self.ocr_enabled = False

    def clean(self, text: str, **kwargs) -> str:
        if not text:
            return ""

        try:
            logger.info(f"Starting PDF extraction, input type: {type(text)}")

            if isinstance(text, str) and text.endswith('.pdf'):
                with open(text, 'rb') as f:
                    pdf_bytes = f.read()
            else:
                if isinstance(text, str):
                    import base64
                    try:
                        pdf_bytes = base64.b64decode(text)
                        logger.info("Decoded base64 PDF")
                    except (TypeError, ValueError, binascii.Error):
                        logger.warning("Failed to decode as base64, treating as raw text")
                        return text
                else:
                    pdf_bytes = text

            self._pdf_bytes = pdf_bytes  # сохраняем для pdfplumber и OCR

            text_content = self._extract_text_and_tables(pdf_bytes)
            return text_content.strip()

        except (OSError, IOError, Exception):
            logger.error("PDF extraction failed", exc_info=True)
            return ""

    def _extract_text_and_tables(self, pdf_bytes: bytes) -> str:
        """Извлекает текст и таблицы с помощью pdfplumber, таблицы → Markdown."""
        content = ""
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Текст страницы
                    # layout=True помогает сохранить "колонки" и переносы для форм/квитанций.
                    # Tolerate: уменьшаем агрессивное склеивание строк.
                    page_text = page.extract_text(layout=True, x_tolerance=1.5, y_tolerance=2.0)
                    if page_text:
                        content += f"\n\n## Страница {page_num}\n\n{page_text}\n\n"

                    # Таблицы со страницы
                    # Настройки table extraction: для типовых PDF форм/квитанций
                    # пытаемся ловить сетку по линиям/пересечениям.
                    table_settings_lines = {
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "edge_min_length": 20,
                        "intersection_tolerance": 3,
                        "min_words_vertical": 1,
                        "min_words_horizontal": 1,
                    }
                    tables = page.extract_tables(table_settings=table_settings_lines)
                    accepted_any = False
                    for table in tables or []:
                        if not table or len(table) < 2:
                            continue
                        # Важно: ничего не отбрасываем. Если таблица "плохая", всё равно сохраняем как TSV —
                        # LLM сможет извлечь важные значения.
                        md_table = self._table_to_markdown(table)
                        if md_table.strip():
                            content += f"\n{md_table}\n\n"
                            if self._table_looks_reasonable(table):
                                accepted_any = True

                    # Если "линии" не дали нормальных таблиц, пробуем стратегию "по тексту"
                    if not accepted_any:
                        table_settings_text = {
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 3,
                            "join_tolerance": 3,
                            "intersection_tolerance": 3,
                            "min_words_vertical": 2,
                            "min_words_horizontal": 2,
                        }
                        tables2 = page.extract_tables(table_settings=table_settings_text)
                        for table in tables2 or []:
                            if not table or len(table) < 2:
                                continue
                            md_table = self._table_to_markdown(table)
                            if md_table.strip():
                                content += f"\n{md_table}\n\n"
            # Важно: для "сканированных" PDF pdfplumber не падает, но возвращает пусто.
            # В этом случае делаем fallback на OCR/текстовый слой.
            if not content.strip():
                logger.info("pdfplumber returned empty content; falling back to OCR/text layer")
                return self._fallback_extraction()
            return content
        except (OSError, IOError, Exception):
            logger.error("pdfplumber extraction failed, falling back to OCR/text layer", exc_info=True)
            return self._fallback_extraction()

    @staticmethod
    def _table_looks_reasonable(table_data: list) -> bool:
        """
        Эвристика качества таблицы из pdfplumber.
        Отбрасываем таблицы, где:
        - слишком мало колонок
        - большинство ячеек пустые
        - есть ячейки, в которые "впихнулась" почти вся страница (очень длинный текст)
        """
        try:
            rows = [r for r in table_data if r]
            if len(rows) < 2:
                return False
            num_cols = max(len(r) for r in rows)
            if num_cols < 2:
                return False

            cells = []
            for r in rows:
                for c in r:
                    if c is None:
                        cells.append("")
                    else:
                        cells.append(str(c).strip())

            if not cells:
                return False

            non_empty = [c for c in cells if c]
            if len(non_empty) / max(1, len(cells)) < 0.25:
                return False

            # Если хотя бы одна ячейка слишком длинная, это почти наверняка "склейка" всей страницы.
            max_len = max((len(c) for c in non_empty), default=0)
            if max_len >= 500:
                return False

            # Слишком много "длинных" ячеек — тоже подозрительно.
            long_cells = sum(1 for c in non_empty if len(c) >= 200)
            if long_cells / max(1, len(non_empty)) > 0.2:
                return False

            return True
        except (TypeError, AttributeError, ValueError):
            return False

    def _fallback_extraction(self) -> str:
        """Ручное извлечение текста (текстовый слой или OCR)."""
        if not self._pdf_bytes:
            return ""
        text_content = ""
        try:
            pdf_reader = PdfReader(BytesIO(self._pdf_bytes))
            # Проверяем, есть ли текстовый слой
            def _page_text(p):
                try:
                    t = p.extract_text()
                    return (t or "").strip()
                except (AttributeError, TypeError):
                    return ""

            has_text = any(_page_text(page) for page in pdf_reader.pages)
            if has_text:
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text_content += f"\n\n## Страница {page_num}\n\n{_page_text(page)}\n\n"
            elif self.ocr_enabled and OCR_AVAILABLE and convert_from_bytes:
                images = convert_from_bytes(
                    self._pdf_bytes,
                    dpi=300,
                    poppler_path=self.poppler_path,
                    fmt="png",
                )
                for page_num, image in enumerate(images, start=1):
                    text_content += f"\n\n## Страница {page_num}\n\n"
                    if pytesseract:
                        # LSTM, авто-сегментация страницы. Оставляем базовые настройки без "магии".
                        page_text = pytesseract.image_to_string(
                            image,
                            lang=self.ocr_language,
                            config="--oem 1 --psm 3",
                        )
                        text_content += self._clean_ocr_text(page_text)
            if not text_content.strip():
                logger.warning(
                    "Fallback extraction produced empty content "
                    f"(ocr_enabled={self.ocr_enabled}, ocr_available={OCR_AVAILABLE})"
                )
            return text_content
        except (OSError, IOError, Exception):
            logger.error("Fallback extraction failed", exc_info=True)
            return ""

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        """
        Лёгкая постобработка OCR-текста:
        - удаляем строки, состоящие только из символов рамок/мусора
        - нормализуем пробелы, не убивая переносы строк
        """
        if not text:
            return ""
        # В превью/ингесте лучше ничего не терять: делаем только мягкую нормализацию пробелов,
        # сохраняя все строки (включая "рамки", т.к. они могут нести смысл в формах).
        out = []
        for raw in text.splitlines():
            # не strip() целиком — важны ведущие пробелы/символы в некоторых формах
            line = raw.rstrip()
            # нормализация "супер-многих" пробелов
            if line and ("  " in line):
                # не трогаем табы, только схлопываем многократные пробелы до одного
                while "  " in line:
                    line = line.replace("  ", " ")
            out.append(line)
        # Убираем только хвостовые пустые строки
        while out and out[-1] == "":
            out.pop()
        return "\n".join(out).strip("\n") + "\n"

    @staticmethod
    def _table_to_markdown(table_data: list) -> str:
        """Конвертирует таблицу в LLM-friendly текст (TSV в code block)."""
        if not table_data:
            return ""
        num_cols = max(len(row) for row in table_data) if table_data else 0
        rows_out = []
        for row in table_data:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    v = str(cell).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
                    cleaned_row.append(" ".join(v.split()))
            if len(cleaned_row) < num_cols:
                cleaned_row.extend([""] * (num_cols - len(cleaned_row)))
            rows_out.append("\t".join(cleaned_row))

        body = "\n".join(rows_out).strip()
        if not body:
            return ""
        return "```tsv\n" + body + "\n```"