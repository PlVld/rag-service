import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass


class LangChainChunker(BaseChunker):
    URL_PATTERN = re.compile(
        r'(?:[a-z][a-z0-9+.-]*://|www\.)\S+?(?=[\s.,!?;:\'"\])]|$)',
        re.IGNORECASE
    )

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._init_splitters()

    def _init_splitters(self):
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n", "\n**", "\n*", "\n", ".\n", "!\n", "?\n", ". ", "! ", "? ",
                ".", "**", "*", ";", " ", ""
            ]
        )
        # Убираем использование MarkdownHeaderTextSplitter так как он создает ненужные поля
        # Вместо этого используем обычный разделитель
        self.markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n", "\n#", "\n##", "\n###", "\n####", "\n", ".\n", "!\n", "?\n", ". ", "! ", "? ",
                ".", "**", "*", ";", " ", ""
            ]
        )
        self.onec_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n#КонецОбласти", "\nКонецПроцедуры", "\nКонецФункции", "\n\n", "\n", ";", " ", ""]
        )

    @staticmethod
    def _protect_urls(text: str) -> str:
        def replace_dots(match):
            return match.group(0).replace('.', '．')
        return LangChainChunker.URL_PATTERN.sub(replace_dots, text)

    @staticmethod
    def _restore_urls(text: str) -> str:
        return text.replace('．', '.')

    def _merge_small_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.min_chunk_size <= 0:
            return chunks
        merged = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            if len(current["text"]) < self.min_chunk_size:
                if merged:
                    prev = merged[-1]
                    prev["text"] += "\n" + current["text"]
                else:
                    if i + 1 < len(chunks):
                        current["text"] += "\n" + chunks[i+1]["text"]
                        i += 1
                        merged.append(current)
                    else:
                        merged.append(current)
            else:
                merged.append(current)
            i += 1
        return merged

    def _split_markdown(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        split_texts = self.markdown_splitter.split_text(text)
        chunks = []
        for text_chunk in split_texts:
            chunk_metadata = metadata.copy()
            chunks.append({"text": text_chunk, "metadata": chunk_metadata})
        return chunks

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        content_type = metadata.get("content_type", "text").lower()

        # Защита URL от разбиения (для текста и markdown)
        if content_type in ('text', 'markdown'):
            text = self._protect_urls(text)

        # Выбор стратегии разбиения
        if content_type == 'markdown':
            chunks = self._split_markdown(text, metadata)
        elif content_type == 'code':
            split_texts = self.onec_splitter.split_text(text)
            chunks = [{"text": t, "metadata": metadata.copy()} for t in split_texts]
        else:  # text
            split_texts = self.recursive_splitter.split_text(text)
            chunks = [{"text": t, "metadata": metadata.copy()} for t in split_texts]

        # Восстановление URL
        for chunk in chunks:
            chunk["text"] = self._restore_urls(chunk["text"])

        # Категории (category_path / categories) добавляются в нормализованный текст через
        # normalize_for_embedding() в documents_upload.py — не дублируем здесь,
        # иначе они появятся дважды: в raw_text и в normalized_text.
        pass  # category_prefix оставлен в metadata для raw_text/поиска

        # Объединение мелких чанков (кроме кода)
        if content_type in ('text', 'markdown') and self.min_chunk_size > 0:
            chunks = self._merge_small_chunks(chunks)

        return chunks


class ChunkerFactory:
    @staticmethod
    def create_chunker(chunker_type: str = "langchain", **kwargs) -> BaseChunker:
        if chunker_type == "langchain":
            return LangChainChunker(**kwargs)
        raise ValueError(f"Unknown chunker type: {chunker_type}")