import logging
from typing import Optional
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient
from app.core.config import settings

router = APIRouter(tags=["Health"])
_qdrant_client: Optional[AsyncQdrantClient] = None
logger = logging.getLogger(__name__)


async def _check_embedding_model_ready() -> bool:
    """
    Проверяет, что модель эмбеддингов загружена и готова к работе.

    Пытается получить закэшированную модель через get_embedding_model().
    Возвращает True, если модель доступна, иначе False.

    Returns:
        bool: True если модель загружена, False в противном случае.
    """
    from app.core.embeddings import get_embedding_model

    try:
        get_embedding_model()
        return True
    except Exception:
        return False


async def _check_qdrant_available() -> bool:
    """
    Проверяет доступность сервера Qdrant.

    Выполняет асинхронный вызов get_collections() для проверки соединения.
    Возвращает True, если сервер отвечает, иначе False.

    Returns:
        bool: True если Qdrant доступен, False в противном случае.
    """
    client = get_client()
    try:
        await client.get_collections()
        return True
    except Exception:
        return False


@router.get("/health", summary="Проверка живости сервиса (liveness)")
async def health_liveness() -> dict:
    """
    Эндпоинт проверки живости (liveness probe).

    Возвращает 200 OK, если процесс приложения запущен и отвечает.
    Не проверяет внешние зависимости (Qdrant, модель эмбеддингов).
    Используется оркестраторами (Kubernetes, Docker) для определения,
    нужно ли перезапустить контейнер.

    Returns:
        JSONResponse: JSON с полем "status": "ok" и HTTP-кодом 200.
    """
    return {"status": "ok"}


@router.get("/health/ready", summary="Проверка готовности сервиса (readiness)")
async def health_readiness() -> JSONResponse:
    """
    Эндпоинт проверки готовности (readiness probe).

    Проверяет доступность всех критических зависимостей:
    - Сервер Qdrant (подключение и ответ на get_collections)
    - Модель эмбеддингов (загружена и готова к кодированию)

    Возвращает 200 OK, если все зависимости доступны, иначе 503 Service Unavailable.
    Используется оркестраторами для определения, готово ли приложение
    принимать трафик.

    Returns:
        JSONResponse: JSON со статусом проверки и списками доступных/недоступных
        зависимостей. HTTP 200 если все ок, 503 если что-то недоступно.
    """
    qdrant_ok = await _check_qdrant_available()
    model_ok = await _check_embedding_model_ready()

    deps = [
        {"name": "qdrant", "status": "ok" if qdrant_ok else "unavailable"},
        {"name": "embedding_model", "status": "ok" if model_ok else "not_loaded"},
    ]

    all_ok = qdrant_ok and model_ok

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "healthy" if all_ok else "degraded",
            "dependencies": deps,
        },
    )

async def get_client_ready() -> bool:
    """
    Возвращает True, если клиент Qdrand подключён и сервер доступен.

    Создает клиент через get_client() и проверяет ответ get_collections().
    Используется как вспомогательная функция для health readiness probe.

    Returns:
        bool: True если Qdrant доступен, False при ошибке подключения.
    """
    return await _check_qdrant_available()


def get_client() -> AsyncQdrantClient:
    """
    Возвращает экземпляр асинхронного клиента Qdrant.

    Реализует паттерн Singleton: клиент создается один раз при первом обращении
    и переиспользуется в последующих вызовах. Это повышает производительность
    и снижает количество соединений.

    Настройки подключения (URL и API-ключ) берутся из конфигурации приложения.

    Возвращает:
        AsyncQdrantClient: Настроенный клиент для работы с Qdrant
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            # Увеличенный тайм-аут для больших batch-операций (загрузка документов с 1000+ чанков).
            # Дефолтный тайм-аут httpx (5 сек) недостаточен для таких операций.
            timeout=120,
        )
    return _qdrant_client


async def close_client() -> None:
    """Закрывает singleton-клиент и сбрасывает кэш (вызывается при остановке приложения)."""
    global _qdrant_client
    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None

