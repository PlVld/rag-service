import os
import logging
import time
import asyncio
from functools import lru_cache
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.core.config import settings
import torch

logger = logging.getLogger(__name__)

# Папка кэша в корне проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

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
    
    # Проверяем наличие blobs (тяжёлые файлы модели)
    blobs_dir = os.path.join(model_dir, "blobs")
    if not os.path.exists(blobs_dir):
        return False
    
    # Проверяем, не пустой ли blobs
    try:
        if not os.listdir(blobs_dir):
            return False
    except OSError:
        return False
    
    # Проверяем наличие snapshots (загруженных файлов модели)
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        return False
    
    # Проверяем, есть ли хотя бы один файл модели в снапшотах
    for root, dirs, files in os.walk(snapshots_dir):
        for f in files:
            if f in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "config.json"]:
                return True
    return False


def initialize_model(online: bool = False) -> SentenceTransformer:
    """
    Инициализирует и кэширует модель эмбеддингов при старте.
    
    Args:
        online: Если True, загружает модель из интернета если её нет в кэше
    
    Returns:
        SentenceTransformer: Загруженная модель
    
    Raises:
        RuntimeError: Если модель не найдена в кэше и online=False
    """
    model_name = settings.embedding_model
    model_path = os.path.join(CACHE_DIR, "models--" + model_name.replace("/", "--"))
    
    logger.info(f"Checking model cache: {model_path}")
    
    if _check_model_exists(model_name):
        logger.info("Model found in cache. Loading...")
        return _load_model(offline=True)
    
    if not online:
        raise RuntimeError(
            f"Model not found in cache: {model_path}. "
            f"Please run with online=True once to download, or set HF_HUB_OFFLINE=0 and restart."
        )
    
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


def _load_model(offline: bool = True) -> SentenceTransformer:
    """Внутренняя функция загрузки модели."""
    use_gpu = settings.use_gpu and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    load_start = time.time()
    
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
        # Исправленная конфигурация для загрузки модели
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": offline,  # Важно для офлайн-режима
            "revision": None,
        }

        # Принудительно отключаем safetensors проверку, если есть проблемы
        if "safetensors" in settings.embedding_model:
            model_kwargs["use_safetensors"] = False

        model = SentenceTransformer(
            settings.embedding_model,
            device=device,
            cache_folder=CACHE_DIR,
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
def get_embedding_model() -> SentenceTransformer:
    """Загружает и кэширует модель эмбеддингов (использует кэш)."""
    return _load_model(offline=True)

def _encode_batch_sync(
    model: SentenceTransformer,
    texts: List[str],
    encode_kwargs: dict,
) -> List[List[float]]:
    """Синхронная обёртка для пакетного кодирования."""
    embeddings = model.encode(texts, **encode_kwargs)
    return [emb.tolist() for emb in embeddings]


def _encode_single_sync(
    model: SentenceTransformer,
    text: str,
    encode_kwargs: dict,
) -> List[float]:
    """Синхронная обёртка для кодирования одного текста."""
    embedding = model.encode(text, **encode_kwargs)
    return embedding.tolist()


async def encode_text(
    text: str,
    task_type: str = "document",
    dimensions: Optional[int] = None,
    model_name: Optional[str] = None,
) -> List[float]:
    """
    Универсальная асинхронная функция векторизации текста.

    Аргументы:
        text (str): Текст для преобразования в вектор
        task_type (str): Тип задачи ("query" или "document") — влияет на нормализацию
        dimensions (Optional[int]): Желаемая размерность эмбеддинга (если модель поддерживает)
        model_name (Optional[str]): Название модели (если отличается от настроенной)

    Возвращает:
        List[float]: Векторное представление текста

    Raises:
        RuntimeError: Если модель эмбеддингов не загрузилась или произошла ошибка кодирования
    """
    if not text:
        return []

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
    model = get_embedding_model()
    return model.get_embedding_dimension()
