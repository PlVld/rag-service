import logging
from typing import Dict, Any, Optional

from .html_cleaner import HTMLCleaner
from .pdf_cleaner import PDFCleaner
from .doc_cleaner import DOCXCleaner
from .russian_cleaner import RussianTextCleaner

logger = logging.getLogger(__name__)


class TextCleanerPipeline:
    """
    Основной класс для очистки текста.
    Сначала конвертирует все форматы в Markdown, затем применяет лингвистическую очистку.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Инициализируем конвертеры
        self.html_converter = HTMLCleaner(
            ignore_links=self.config.get('ignore_links', False),
            ignore_images=self.config.get('ignore_images', True)
        )
        # MarkdownCleaner не используем для конвертации, т.к. он удаляет разметку
        self.pdf_converter = PDFCleaner(
            ignore_images=self.config.get('ignore_images', True)
        )
        self.docx_converter = DOCXCleaner(
            ignore_images=self.config.get('ignore_images', True)
        )
        self.russian_cleaner = RussianTextCleaner(
            remove_punctuation=self.config.get('remove_punctuation', False),
            remove_stopwords=self.config.get('remove_stopwords', True),
            lowercase=self.config.get('lowercase', True),
            lemmatize=self.config.get('lemmatize', False)
        )

    def convert_to_markdown(self, text: str, source_format: str = 'text') -> str:
        """Конвертирует в Markdown без лишней очистки."""
        if not text:
            return ""
        if source_format == 'html':
            return self.html_converter.clean(text)
        if source_format == 'markdown':
            return text
        if source_format == 'pdf':
            return self.pdf_converter.clean(text)
        if source_format == 'docx':
            return self.docx_converter.clean(text)
        if source_format == 'xlsx':
            return self._xlsx_to_markdown(text)
        if source_format == 'code':
            lang = self._detect_language(text)
            return f"```{lang}\n{text}\n```"
        return text

    @staticmethod
    def _xlsx_to_markdown(file_path: str) -> str:
        """Конвертирует XLSX файл в Markdown (каждый лист → таблица)."""
        try:
            import pandas as pd
            xlsx = pd.ExcelFile(file_path)
            content = ""
            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
                # Превращаем DataFrame в Markdown-таблицу
                markdown_table = df.to_markdown(index=False, tablefmt="pipe")
                content += f"## Лист: {sheet_name}\n\n{markdown_table}\n\n"
            return content
        except ImportError:
            logger.error("pandas not installed, cannot convert XLSX")
            return ""
        except Exception as e:
            logger.error(f"XLSX conversion failed: {e}")
            return ""

    @staticmethod
    def _detect_language(code: str) -> str:
        """Простейшее определение языка кода по ключевым словам."""
        if 'процедура' in code.lower() or 'функция' in code.lower():
            return '1c'
        if 'def ' in code or 'import ' in code:
            return 'python'
        if 'function ' in code or 'var ' in code:
            return 'javascript'
        return ''