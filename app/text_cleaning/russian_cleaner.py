from stop_words import get_stop_words
from .base import BaseCleaner


class RussianTextCleaner(BaseCleaner):
    """
    Лингвистическая очистка для русского языка.
    Удаляет стоп-слова, приводит к нижнему регистру, убирает пунктуацию (опционально).
    """

    def __init__(
        self,
        remove_punctuation: bool = False,
        remove_stopwords: bool = True,
        lowercase: bool = True,
        lemmatize: bool = False
    ):
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        
        if remove_stopwords:
            # Получить русские стоп-слова
            self.stop_words = set(get_stop_words('ru'))
        else:
            self.stop_words = set()

    def clean(self, text: str, **kwargs) -> str:
        if not text:
            return ""
        
        result = text
        
        # Удаление лишних пробелов (spaces=True из ru_text_cleaner)
        result = ' '.join(result.split())
        
        # Приведение к нижнему регистру
        if self.lowercase:
            result = result.lower()
        
        # Удаление стоп-слов
        if self.remove_stopwords:
            words = result.split()
            filtered_words = [w for w in words if w not in self.stop_words]
            result = ' '.join(filtered_words)
        
        # Удаление пунктуации (простая реализация)
        if self.remove_punctuation:
            import string
            result = result.translate(str.maketrans('', '', string.punctuation))
        
        return result.strip()