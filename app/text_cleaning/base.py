from abc import ABC, abstractmethod

class BaseCleaner(ABC):
    """Базовый класс для всех очистителей текста."""

    @abstractmethod
    def clean(self, text: str, **kwargs) -> str:
        """Очищает текст и возвращает результат."""
        pass


class CompositeCleaner(BaseCleaner):
    """Композитный очиститель, применяющий несколько очистителей последовательно."""

    def __init__(self, cleaners: list[BaseCleaner]):
        self.cleaners = cleaners

    def clean(self, text: str, **kwargs) -> str:
        for cleaner in self.cleaners:
            text = cleaner.clean(text, **kwargs)
        return text
