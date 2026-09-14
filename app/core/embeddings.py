import os
import logging
import time
import asyncio
from functools import lru_cache
from typing import List, Optional, Union

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Максимальный размер батча для одного запроса к API (TEI по умолчанию ограничивает батч)
REMOTE_EMBED_BATCH_SIZE = 32


# =============================================================================
# Удалённый режим (text-embeddings-inference и совместимые API)
# =============================================================================
def remote_mode_enabled() -> bool:
    """True, если эмбеддинги считаются через удалённый сервис."""
    return bool(settings.embeddings_api_url)


_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            base_url=settings.embeddings_api_url,
            timeout=httpx.Timeout(settings.embeddings_api_timeout),
        )
    return _http_client


async def close_http_client() -> None:
    """Закрывает HTTP-клиент (вызывается при остановке приложения)."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


async def _remote_embed(texts: List[str]) -> List[List[float]]:
    """Кодирует список текстов через /embed, разбивая на батчи."""
    client = _get_http_client()
    result: List[List[float]] = []
    for i in range(0, len(texts), REMOTE_EMBED_BATCH_SIZE):
        chunk = texts[i:i + REMOTE_EMBED_BATCH_SIZE]
        resp = await client.post("/embed", json={"inputs": chunk})
        resp.raise_for_status()
        result.extend(resp.json())
    return result


async def check_embedding_ready() -> bool:
    """Проверяет готовность модели эмбеддингов (удалённой или локальной)."""
    if remote_mode_enabled():
        try:
            resp = await _get_http_client().get("/health")
            return resp.status_code == 200
        except Exception:
            return False
    try:
        get_embedding_model()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _remote_dimension() -> int:
    """Определяет размерность вектора пробным запросом (кэшируется)."""
    with httpx.Client(
        base_url=settings.embeddings_api_url,
        timeout=settings.embeddings_api_timeout,
    ) as client:
        resp = client.post("/embed", json={"inputs": ["dimension probe"]})
        resp.raise_for_status()
        return len(resp.json()[0])


# =============================================================================
# Локальный режим (sentence-transformers). Используется, когда
# EMBEDDINGS_API_URL не задан (например, запуск вне Docker).
# =============================================================================
CACHE_DIR = os.environ.get("SENTENCE_TRANSFORMERS_HOME",
               os.environ.get("HF_HOME",
               os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model_cache")))

# Полностью отключаем все сетевые функции Hugging Face
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Без прогресс-баров
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Подавляем предупреждения
os.environ["HF_TOKEN"] = ""                    # Пустой токен, чтобы не пытался аутентифицироваться

# Словарь задач для разных моделей (при необходимости расширять)
TASK_MAP = {
    "jinaai/jina-embeddings-v3": {
        "query": "retrieval.query",
        "document": "retrieval.passage",
    },
    "BAAI/bge-m3": {
        "query": None,
        "document": None,
    },
    "default": {
        "query": None,
        "document": None,
    }
}


def _check_model_exists(model_name: str) -> bool:
    """Проверяет наличие модели в кэше."""
    model_dir = os.path.join(CACHE_DIR, "models--" + model_name.replace("/", "--"))
    if not os.path.exists(model_dir):
        return False

    # Проверяем наличие snapshots (загруженных файлов модели).
    # Blobs могут быть пустыми: на Windows huggingface_hub без симлинков
    # кладёт файлы прямо в snapshots.
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        return False

    # Проверяем, есть ли хотя бы один файл модели в снапшотах
    for root, dirs, files in os.walk(snapshots_dir):
        for f in files:
            if f in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "config.json"]:
                return True
    return False


def initialize_model(online: bool = False):
    """
    Инициализирует модель эмбеддингов при старте.

    В удалённом режиме проверяет доступность API и возвращает None.
    В локальном — загружает модель из кэша (или из интернета, если online=True).

    Raises:
        RuntimeError: Если модель не найдена в кэше и online=False,
                      или удалённый API недоступен.
    """
    if remote_mode_enabled():
        with httpx.Client(base_url=settings.embeddings_api_url, timeout=10.0) as client:
            resp = client.get("/health")
            if resp.status_code != 200:
                raise RuntimeError(f"Embeddings API is not ready: HTTP {resp.status_code}")
        logger.info(f"Embeddings API is ready: {settings.embeddings_api_url}")
        return None

    from sentence_transformers import SentenceTransformer

    if not _check_model_exists(settings.embedding_model) and not online:
        raise RuntimeError(
            f"Model not found in cache: {os.path.join(CACHE_DIR, 'models--' + settings.embedding_model.replace('/', '--'))}. "
            f"Please run with online=True once to download, or set HF_HUB_OFFLINE=0 and restart."
        )

    if _check_model_exists(settings.embedding_model):
        logger.info("Model found in cache. Loading...")
        return _load_model(offline=True)

    logger.info("Model not found in cache. Downloading from Hugging Face...")
    logger.info(f"Cache directory: {CACHE_DIR}")

    # Временно включаем онлайн режим для загрузки
    old_offline = os.environ.get("HF_HUB_OFFLINE", "0")
    old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE", "0")

    try:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

        logger.info("Downloading model (this may take a while)...")
        model = _load_model(offline=False)

        logger.info("Model downloaded and cached successfully!")
        return model
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise
    finally:
        # Восстанавливаем офлайн режим
        os.environ["HF_HUB_OFFLINE"] = old_offline
        os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline


def _load_model(offline: bool = True):
    """Внутренняя функция загрузки модели (локальный режим)."""
    import torch
    from sentence_transformers import SentenceTransformer

    use_gpu = settings.use_gpu and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    load_start = time.time()

    os.makedirs(CACHE_DIR, exist_ok=True)

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

    logger.info(f"Loading embedding model on {device}: {settings.embedding_model}")
    logger.info(f"Model cache directory: {CACHE_DIR}")
    logger.info(f"Offline mode: {offline}")

    try:
        # Устанавливаем HF_HOME для правильного кэширования (современный API)
        os.environ["HF_HOME"] = CACHE_DIR

        model_kwargs = {
            "local_files_only": offline,  # Важно для офлайн-режима
            "revision": None,
        }

        # Принудительно отключаем safetensors проверку, если есть проблемы
        if "safetensors" in settings.embedding_model:
            model_kwargs["use_safetensors"] = False

        model = SentenceTransformer(
            settings.embedding_model,
            device=device,
            model_kwargs=model_kwargs,
        )
        load_time = time.time() - load_start
        logger.info(f"[PERF] Model loaded in {load_time:.3f}s")
        return model
    except Exception as e:
        # Дополнительная проверка для диагностики
        model_path = os.path.join(CACHE_DIR, "models--" + settings.embedding_model.replace("/", "--"))
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model not found in cache: {model_path}. Please ensure the model is pre-downloaded.") from e

        config_path = os.path.join(model_path, "snapshots")
        if os.path.exists(model_path) and not os.path.exists(config_path):
            logger.warning(f"Model directory exists but no snapshots found: {model_path}")

        raise RuntimeError(f"Failed to load model from cache even though path exists: {model_path}") from e


@lru_cache(maxsize=1)
def get_embedding_model():
    """Загружает и кэширует модель эмбеддингов (локальный режим)."""
    if remote_mode_enabled():
        raise RuntimeError(
            "Local embedding model is disabled: EMBEDDINGS_API_URL is set. "
            "Use check_embedding_ready()/encode_text() instead."
        )
    return _load_model(offline=True)


def _encode_batch_sync(
    model,
    texts: List[str],
    encode_kwargs: dict,
) -> List[List[float]]:
    """Синхронная обёртка для пакетного кодирования."""
    embeddings = model.encode(texts, **encode_kwargs)
    return [emb.tolist() for emb in embeddings]


def _encode_single_sync(
    model,
    text: str,
    encode_kwargs: dict,
) -> List[float]:
    """Синхронная обёртка для кодирования одного текста."""
    embedding = model.encode(text, **encode_kwargs)
    return embedding.tolist()


# =============================================================================
# Публичный API (единый для обоих режимов)
# =============================================================================
async def encode_text(
    text: Union[str, List[str]],
    task_type: str = "document",
    dimensions: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Union[List[float], List[List[float]]]:
    """
    Универсальная асинхронная функция векторизации текста.

    Аргументы:
        text (str | List[str]): Текст или список текстов для преобразования в вектор
        task_type (str): Тип задачи ("query" или "document") — влияет на нормализацию
        dimensions (Optional[int]): Желаемая размерность эмбеддинга (если модель поддерживает)
        model_name (Optional[str]): Название модели (если отличается от настроенной)

    Возвращает:
        List[float] | List[List[float]]: Вектор(ы) представления текста

    Raises:
        RuntimeError: Если модель эмбеддингов не загрузилась или произошла ошибка кодирования
    """
    if not text:
        return []

    if remote_mode_enabled():
        start_time = time.time()
        try:
            if isinstance(text, list):
                embeddings = await _remote_embed(text)
                logger.info(f"[PERF] encode_text: batch of {len(text)} texts in {time.time() - start_time:.3f}s (remote)")
                return embeddings
            embedding = (await _remote_embed([text]))[0]
            logger.info(f"[PERF] encode_text: {time.time() - start_time:.3f}s (remote), task={task_type}, length={len(text)}")
            return embedding
        except Exception as e:
            error_msg = f"Failed to encode text via embeddings API: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    start_time = time.time()
    logger.info(f"[PERF] encode_text start: task={task_type}, length={len(text) if isinstance(text, str) else len(text)}")

    try:
        load_model_start = time.time()
        model = get_embedding_model()
        load_model_time = time.time() - load_model_start
        logger.info(f"[PERF] Model lookup/load: {load_model_time:.3f}s")
    except Exception as e:
        error_msg = f"Failed to load embedding model: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    effective_model_name = model_name or settings.embedding_model

    try:
        task_config = TASK_MAP.get(effective_model_name, TASK_MAP["default"])
        model_task = task_config.get(task_type)

        encode_kwargs = {}
        if model_task:
            encode_kwargs['task'] = model_task
        if dimensions:
            encode_kwargs['dimensions'] = dimensions

        # Поддержка пакетной обработки
        if isinstance(text, list):
            encode_start = time.time()
            embeddings = await asyncio.to_thread(_encode_batch_sync, model, text, encode_kwargs)
            encode_time = time.time() - encode_start
            total_time = time.time() - start_time
            logger.info(f"[PERF] encode_text: batch of {len(text)} texts in {total_time:.3f}s (encode: {encode_time:.3f}s)")
            return embeddings

        encode_start = time.time()
        embedding = await asyncio.to_thread(_encode_single_sync, model, text, encode_kwargs)
        encode_time = time.time() - encode_start
        total_time = time.time() - start_time
        logger.info(f"[PERF] encode_text: {total_time:.3f}s (encode: {encode_time:.3f}s), task={task_type}, length={len(text)}")
        return embedding

    except Exception as e:
        error_msg = f"Failed to encode text: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def get_embedding_dimension() -> int:
    """Возвращает полную размерность векторов текущей модели (для создания коллекций)."""
    if remote_mode_enabled():
        return _remote_dimension()
    return get_embedding_model().get_embedding_dimension()
