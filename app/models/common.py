from typing import Optional, Union, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, model_validator, model_serializer
import json

class DocumentCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Пример текста документа для векторизации. "
                            "Это может быть статья, инструкция, отчёт или любой другой документ.",
                    "payload": {
                        "source_format": "text",
                        "author": "Иванов И.И."
                    },
                    "category_path": ["Документация", "Инструкции"],
                    "filename": "instruction.txt",
                    "title": "Инструкция пользователя",
                    "version": 1
                }
            ]
        }
    )
    text: str = Field(
        ...,
        description="Текст документа для векторизации",
        min_length=1,
        examples=["Пример текста документа для векторизации."]
    )
    payload: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Произвольные метаданные документа (source_format, author и т.д.)",
        examples=[{"source_format": "text", "author": "Иванов И.И."}]
    )
    category_path: Optional[List[str]] = Field(
        default=None,
        description="Иерархия категорий в виде списка строк",
        examples=[["Документация", "Инструкции"]]
    )
    filename: Optional[str] = Field(
        default=None,
        description="Имя файла (используется для генерации source_id, если не указан)",
        examples=["instruction.txt"]
    )
    title: Optional[str] = Field(
        default=None,
        description="Заголовок документа (используется для генерации source_id)",
        examples=["Инструкция пользователя"]
    )
    version: Optional[int] = Field(
        default=1,
        description="Версия документа (автоинкремент при обновлении)",
        ge=1,
        examples=[1]
    )


class DocumentsUploadRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "documents": [
                        {
                            "text": "Текст первого документа для векторизации и поиска.",
                            "payload": {"source_format": "text"},
                            "category_path": ["Документация", "API"],
                            "filename": "api_doc.txt",
                            "version": 1
                        },
                        {
                            "text": "Текст второго документа с описанием методов обработки данных.",
                            "payload": {"source_format": "text"},
                            "category_path": ["Документация", "Руководства"],
                            "filename": "guide.txt",
                            "version": 1
                        }
                    ],
                    "collection_name": "documents"
                }
            ]
        }
    )
    documents: List[DocumentCreate] = Field(
        ...,
        description="Список документов для загрузки",
        min_length=1,
        examples=[
            [
                {
                    "text": "Текст документа для загрузки.",
                    "payload": {"source_format": "text"},
                    "category_path": ["Документация"],
                    "filename": "doc.txt",
                    "version": 1
                }
            ]
        ]
    )
    collection_name: str = Field(
        ...,
        description="Имя коллекции в Qdrant для загрузки документов",
        examples=["documents"]
    )


class DocumentsUploadResponse(BaseModel):
    status: str
    uploaded: int
    ids: List[Union[int, str]] = Field(..., description="Список идентификаторов загруженных документов")
    message: Optional[str] = None


class DocumentSearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query_text": "Как настроить векторный поиск по документам?",
                    "collection_name": "documents",
                    "limit": 10,
                    "filter": {"source_format": "text"},
                    "include_old_versions": False,
                    "max_text_length": 2000,
                    "group": True,
                    "payload_fields": ["category_path", "original_filename"]
                }
            ]
        }
    )
    query_text: str = Field(
        ...,
        description="Текст запроса для семантического поиска",
        examples=["Как настроить векторный поиск по документам?"]
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Условия фильтрации по полям payload (например, source_format, category_path). Может быть строкой JSON или объектом.",
        examples=[{"source_format": "text", "is_latest": True}]
    )
    limit: int = Field(
        default=10,
        description="Максимальное количество возвращаемых результатов",
        ge=1,
        le=1000,
        examples=[10]
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Имя коллекции в Qdrant для поиска. Если не указано, поиск выполняется по всем коллекциям, кроме служебной коллекции категорий.",
        examples=["documents"]
    )
    include_old_versions: bool = Field(
        default=False,
        description="Включать ли устаревшие версии документов (is_latest=False)",
        examples=[False]
    )
    max_text_length: Optional[int] = Field(
        default=None,
        description="Максимальная длина текста в результате (null или 0 = без ограничений). Если задано положительное число, текст будет обрезан до указанной длины.",
        ge=0,
        examples=[2000, None]
    )
    group: bool = Field(
        default=True,
        description="Группировать ли результаты по category_path (true) или возвращать отдельные чанки (false)",
        examples=[True, False]
    )
    payload_fields: Optional[List[str]] = Field(
        default=None,
        description="Список полей payload, которые нужно вернуть. Если не указано, payload не возвращается.",
        examples=[["category_path", "original_filename", "source_id"]]
    )
    
    @model_validator(mode='before')
    @classmethod
    def validate_filter(cls, values):
        """Конвертирует filter из строки JSON в dict, если необходимо."""
        if isinstance(values, dict):
            filter_val = values.get('filter')
            if isinstance(filter_val, str):
                try:
                    values['filter'] = json.loads(filter_val)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON in filter field")
        return values


class DocumentSearchResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "category_path": "Документация / Инструкции",
                    "score": 0.85,
                    "document": "Текст чанка или сгруппированного документа",
                    "payload": {"original_filename": "doc.txt"},
                    "collection_name": "documents"
                }
            ]
        }
    )
    category_path: Optional[str] = Field(default=None, description="Путь категории документа")
    score: float
    document: str
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Метаданные чанка. Включается только если payload_fields указан.")
    collection_name: str = Field(..., description="Имя коллекции, в которой найден чанк")
    
    @model_serializer
    def serialize_model(self):
        """Сериализует модель, исключая поля со значением None."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DocumentsSearchResponse(BaseModel):
    results: List[DocumentSearchResult]


class VersionResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "version": 1,
                    "raw_text": "Полный текст версии документа с соседними чанками.",
                    "created_at": "2024-01-15T10:30:00Z",
                    "file_path": "/path/to/file.txt",
                    "is_latest": True,
                    "total_chunks": 5,
                    "first_chunk_index": 0,
                    "last_chunk_index": 4
                }
            ]
        }
    )
    version: int
    raw_text: str
    created_at: Optional[str] = None
    file_path: Optional[str] = None
    is_latest: bool
    total_chunks: int
    first_chunk_index: int
    last_chunk_index: int


class GroupedDocumentResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "source_id": "abc123-def456",
                    "category_path": "Документация / Инструкции",
                    "original_filename": "instruction.txt",
                    "content_type": "text",
                    "collection_name": "documents",
                    "score": 0.85,
                    "chunk_score": 0.75,
                    "versions": [
                        {
                            "version": 1,
                            "raw_text": "Текст документа...",
                            "is_latest": True,
                            "total_chunks": 3,
                            "first_chunk_index": 0,
                            "last_chunk_index": 2
                        }
                    ]
                }
            ]
        }
    )
    source_id: str
    category_path: str
    original_filename: Optional[str] = None
    content_type: Optional[str] = None
    collection_name: str
    score: float
    chunk_score: float = Field(default=0.0, description="Максимальный score чанка в категории (используется для расчета итогового score)")
    versions: List[VersionResult]


class GroupedDocumentsSearchResponse(BaseModel):
    results: List[GroupedDocumentResult]


class CategorySearchGroupedResponse(BaseModel):
    """Response model for grouped category search results."""
    results: Dict[str, List[dict]]  # Словарь where keys - коллекции, values - списки категорий в формате dict
