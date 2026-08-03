from mdclense.parser import MarkdownParser
from .base import BaseCleaner


class MarkdownCleaner(BaseCleaner):
    """Очищает Markdown-текст, преобразуя в plain text."""

    def __init__(self):
        self.parser = MarkdownParser()

    def clean(self, text: str, **kwargs) -> str:
        if not text:
            return ""
        return self.parser.parse(text).strip()
