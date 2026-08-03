from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

class Settings(BaseSettings):
    # Qdrant
    qdrant_url: str = Field(default="http://qdrant:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", validation_alias="QDRANT_API_KEY")
    rag_service_api_key: str = Field(default="", validation_alias="RAG_SERVICE_API_KEY")
    # Comma-separated list of MCP tools that are allowed to be invoked via /mcp.
    # Examples (currently available tools are listed in app/mcp_server.py):
    #   - search_documents_tool
    #   - search_categories_tool
    # Leave empty to allow all registered tools.
    allowed_mcp_tools: str = Field(default="", validation_alias="ALLOWED_MCP_TOOLS", description="Comma-separated allowed MCP tools")
    default_collection: str = Field(default="documents", validation_alias="DEFAULT_COLLECTION")

    # MCP Authentication
    mcp_auth_enabled: bool = Field(default=True, validation_alias="MCP_AUTH_ENABLED", description="Enable authentication for MCP endpoints")

    @property
    def allowed_mcp_tools_set(self) -> set:
        """Returns a set of allowed MCP tool names parsed from allowed_mcp_tools.

        If the configuration value is empty, returns an empty set meaning "allow all".
        """
        raw = (self.allowed_mcp_tools or "").strip()
        if not raw:
            return set()
        return {p.strip() for p in raw.split(",") if p.strip()}
    semantic_weight: float = Field(default=0.3, validation_alias="SEMANTIC_WEIGHT", ge=0, le=1)
    category_weight: float = Field(default=0.7, validation_alias="CATEGORY_WEIGHT", ge=0, le=1)
    enable_weighted_rrf: bool = Field(default=False, validation_alias="ENABLE_WEIGHTED_RRF")

    # Weighted score parameters for grouped_search
    # Итоговый score = doc_score * document_weight + category_score * category_weight
    document_weight: float = Field(default=0.4, validation_alias="DOCUMENT_WEIGHT", ge=0, le=1)
    category_weight_final: float = Field(default=0.6, validation_alias="CATEGORY_WEIGHT_FINAL", ge=0, le=1)

    # Embedding model
    embedding_model: str = Field(
        default="BAAI/bge-m3",  # или другая модель
        validation_alias="EMBEDDING_MODEL"
    )
    use_gpu: bool = Field(default=True, validation_alias="USE_GPU")

    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # Service
    service_host: str = Field(default="0.0.0.0", validation_alias="SERVICE_HOST")
    service_port: int = Field(default=8000, validation_alias="SERVICE_PORT", ge=1, le=65535)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper().strip()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL '{v}'. Must be one of: {', '.join(valid_levels)}"
            )
        return v_upper

    def configure_logging(self):
        """Configure application-wide logging based on log_level setting."""
        import logging
        log_level = getattr(logging, self.log_level)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Search parameters
    search_limit: int = Field(default=10, validation_alias="SEARCH_LIMIT", ge=1, le=1000)

    # Upload parameters
    upload_batch_size: int = Field(default=32, validation_alias="UPLOAD_BATCH_SIZE", ge=1)
    max_file_size_mb: int = Field(default=50, validation_alias="MAX_FILE_SIZE_MB", ge=1)
    allowed_mime_types: str = Field(
        default="text/plain,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/html,text/markdown",
        validation_alias="ALLOWED_MIME_TYPES"
    )

    # Chunking parameters
    chunk_size: int = Field(default=512, validation_alias="CHUNK_SIZE", ge=1)
    chunk_overlap: int = Field(default=50, validation_alias="CHUNK_OVERLAP", ge=0)

    # Category settings
    category_collection: str = Field(default="categories", validation_alias="CATEGORY_COLLECTION")
    category_boost_factor: float = Field(default=0.5, validation_alias="CATEGORY_BOOST_FACTOR", ge=0.0, le=10.0)
    category_boost_mode: str = Field(default="multiply", validation_alias="CATEGORY_BOOST_MODE")

    # Docling settings
    use_docling: bool = Field(default=True, validation_alias="USE_DOCLING")
    docling_ocr_engine: str = Field(default="tesseract", validation_alias="DOCLING_OCR_ENGINE")
    docling_do_ocr: bool = Field(default=True, validation_alias="DOCLING_DO_OCR")
    docling_image_description_model: str = Field(default="", validation_alias="DOCLING_IMAGE_DESCRIPTION_MODEL")
    docling_images_scale: float = Field(default=1.0, validation_alias="DOCLING_IMAGES_SCALE", gt=0, le=4.0)

    model_config = SettingsConfigDict(extra="allow", env_file=".env", env_file_encoding="utf-8")

    @field_validator("rag_service_api_key")
    @classmethod
    def validate_api_key_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "RAG_SERVICE_API_KEY must not be empty. "
                "Please set a valid API key in the .env file or environment variables."
            )
        return v.strip()

settings = Settings()