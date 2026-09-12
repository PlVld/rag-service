import asyncio
import logging
import time
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from app.api.health import get_client

from starlette.middleware.base import BaseHTTPMiddleware
from app.api import documents, health, categories, files, admin, docs
from app.core.config import settings
from app.mcp import register_mcp_routes, register_mcp_middlewares, mcp_app

# Настройка логирования из конфигурации
settings.configure_logging()
# Уменьшаем уровень для шумных библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

diag_logger = logging.getLogger("rag_service.diagnostics")

# --- Middleware для обработки "Expect: 100-continue" + "Transfer-Encoding: chunked" ---
_BODYLESS_METHODS = {"GET", "HEAD", "OPTIONS", "DELETE"}


class ConsumeRequestBodyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if (request.method in _BODYLESS_METHODS
                and "chunked" in request.headers.get("transfer-encoding", "")):
            await request.body()
        return await call_next(request)


class RequestDiagnosticsMiddleware(BaseHTTPMiddleware):
    # Чувствительные ключи заголовков/куки, которые нужно маскировать
    _SENSITIVE_HEADERS = {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "proxy-authorization",
        "authenticate-info",
        "proxy-authenticate-info",
    }

    async def dispatch(self, request: Request, call_next):
        client = request.client
        client_info = f"{client.host}:{client.port}" if client else "unknown"
        method = request.method
        url = str(request.url)

        # Маскируем чувствительные заголовки и куки
        headers = {
            k: "<REDACTED>" if k.lower() in self._SENSITIVE_HEADERS else v
            for k, v in request.headers.items()
        }

        diag_logger.info(
            "Incoming request: client=%s method=%s url=%s headers=%s",
            client_info, method, url, headers,
        )

        start = time.time()
        try:
            response = await call_next(request)
            elapsed = time.time() - start
            diag_logger.info(
                "Response: client=%s method=%s url=%s status=%s time=%.3fs",
                client_info, method, url, response.status_code, elapsed,
            )
            return response
        except Exception as exc:
            elapsed = time.time() - start
            diag_logger.error(
                "Request failed: client=%s method=%s url=%s error=%s time=%.3fs",
                client_info, method, url, exc, elapsed,
            )
            raise


app = FastAPI(
    title="Universal Document Vector Search Service",
    description="Микросервис для векторного поиска по документам с поддержкой метаданных, категорий и версионирования",
    version="1.2.0"
)

app.add_middleware(RequestDiagnosticsMiddleware)
app.add_middleware(ConsumeRequestBodyMiddleware)

# Подключаем роутеры
app.include_router(documents.upload_router)
app.include_router(documents.search_router)
app.include_router(health.router)
app.include_router(categories.router)
app.include_router(files.router)
app.include_router(admin.router)
app.include_router(docs.router)

# Раздача изображений, извлечённых из документов.
# Доступ без токена: рендерер Markdown или модель не подставят заголовок Authorization
# при загрузке картинки. Листинга каталогов у StaticFiles нет, а имена файлов содержат
# хеш содержимого и не угадываются.
_media_root = Path(settings.media_dir)
_media_root.mkdir(parents=True, exist_ok=True)
app.mount(settings.media_url_prefix, StaticFiles(directory=_media_root), name="media")

# --- MCP: middlewares и маршруты прокси (логика в app/mcp/) ---
register_mcp_middlewares(app)
register_mcp_routes(app)
# Монтируем mcp_app для fallback (не используется напрямую, но нужно для initialize/ping)
app.mount("/_internal_mcp", mcp_app)


# --- Корневой эндпоинт (только один) ---
@app.get("/")
async def root():
    return {"message": "Document Vector Search Service is running", "docs": "/docs"}


# --- Lifespan event handlers (instead of deprecated on_event) ---
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup
    settings.configure_logging()
    diag_logger.info("Starting up RAG service...")

    # Загружаем модель эмбеддингов при старте
    # Сначала пробуем из кэша, если нет — скачиваем
    try:
        from app.core.embeddings import initialize_model
        model = await asyncio.to_thread(initialize_model, online=False)
        diag_logger.info(f"Embedding model loaded from cache: {type(model).__name__}")
    except RuntimeError as e:
        if "not found in cache" in str(e):
            diag_logger.warning("Model not in cache. Downloading from Hugging Face...")
            try:
                from app.core.embeddings import initialize_model
                model = await asyncio.to_thread(initialize_model, online=True)
                diag_logger.info(f"Embedding model downloaded and initialized: {type(model).__name__}")
            except Exception as download_err:
                diag_logger.error(f"Failed to download model: {download_err}")
                diag_logger.error("Set HF_HUB_OFFLINE=0 and restart to download the model.")
                raise RuntimeError(f"Cannot initialize embedding model. Neither cache nor download available.") from download_err
        else:
            diag_logger.error(f"Failed to initialize embedding model: {e}")
            raise

    diag_logger.info("MCP tools are ready via custom proxy.")
    yield
    # Shutdown
    from app.api.health import close_client
    await close_client()
    diag_logger.info("Qdrant client closed.")


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=False
    )
