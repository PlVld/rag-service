import html2text
from .base import BaseCleaner


class HTMLCleaner(BaseCleaner):
    """Очищает HTML-текст, преобразуя в читаемый текст."""

    def __init__(self, ignore_links: bool = False, ignore_images: bool = True):
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = ignore_links
        self.converter.ignore_images = ignore_images
        self.converter.body_width = 0  # отключить перенос строк
        self.converter.quote = False # не добавлять кавычки

    def clean(self, text: str, **kwargs) -> str:
        if not text:
            return ""
        return self.converter.handle(text).strip()