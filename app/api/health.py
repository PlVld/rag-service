import logging
from fastapi import APIRouter
from qdrant_client import QdrantClient
from app.core.config import settings

router = APIRouter(tags=["Health"])
_qdrant_client = None
logger = logging.getLogger(__name__)

def get_client() -> QdrantClient:
    """
    Возвращает экземпляр клиента Qdrant для взаимодействия с векторной базой данных.
    
    Реализует паттерн Singleton: клиент создается один раз при первом обращении
    и переиспользуется в последующих вызовах. Это повышает производительность
    и снижает количество соединений.
    
    Настройки подключения (URL и API-ключ) берутся из конфигурации приложения.
    
    Возвращает:
        QdrantClient: Настроенный клиент для работы с Qdrant
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            # Увеличенный тайм-аут для больших batch-операций (загрузка документов с 1000+ чанков).
            # Дефолтный тайм-аут httpx (5 сек) недостаточен для таких операций.
            timeout=120,
        )
    return _qdrant_client

