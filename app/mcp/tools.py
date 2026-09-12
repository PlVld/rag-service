# app/mcp/tools.py
# MCP-инструменты и кэш информации о коллекциях.
#
# Инструменты регистрируются в явном реестре TOOLS ниже.
# Чтобы скрыть или отключить инструменты, настройте ALLOWED_MCP_TOOLS в окружении
# (имена через запятую). См. app/core/config.py.

import logging
import time

from app.api.health import get_client
from app.mcp_server import (
    SearchDocumentsInput,
    SearchCategoriesInput,
    GetCategoryHierarchyInput,
    _search_documents_internal,
    _search_categories_internal,
    _get_category_hierarchy_internal,
)

logger = logging.getLogger(__name__)


async def search_documents_tool(input_data: SearchDocumentsInput) -> str:
    """Поиск соответствующих тексту запроса чанков.

    По умолчанию результаты группируются по category_path (group=true).
    Чтобы получить отдельные чанки, установите group=false.

    Изображения, извлечённые из исходных документов (PDF, DOCX, HTML).
    Текст чанка — это Markdown, и в нём могут встречаться ссылки на картинки,
    например:
        ![Image](/media/<source_id>/<hash>/images/image_000000_ab12cd34.png)
    Картинки раздаёт этот же RAG-сервис по HTTP, токен для их загрузки не нужен
    (StaticMount настроен без аутентификации).
    Чтобы вставить картинку в ответ пользователю, используйте абсолютный URL:
    подставьте базовый адрес RAG-сервиса перед путём `/media/...`
    (например, `http://localhost:8000/media/...`).
    """
    result = await _search_documents_internal(
        query_text=input_data.query_text,
        collection_name=input_data.collection_name,
        limit=input_data.limit,
        filter_criteria=input_data.filter_criteria,
        include_old_versions=input_data.include_old_versions,
        max_text_length=input_data.max_text_length,
        group=input_data.group,
    )

    if result["success"]:
        results = result["data"]["results"]
        if not results:
            return "По вашему запросу ничего не найдено."

        output = f"Найдено {len(results)} результатов:\n\n"
        for i, r in enumerate(results, 1):
            # Используем подготовленный snippet из mcp_server (уже с учетом max_text_length)
            snippet = r["document"]
            output += f"{i}. [score={r['score']:.3f}] {snippet}\n\n"
        return output
    else:
        return f"Ошибка поиска: {result['error']['message']}"


async def search_categories_tool(input_data: SearchCategoriesInput) -> str:
    """Поиск категорий документов, соответствующих текстовому запросу.
    Категории возвращаются сгруппированными по коллекциям.
    """
    from app.api.categories import search_categories_by_collections

    result = await _search_categories_internal(
        query_text=input_data.query_text,
        limit=input_data.limit,
        fields=input_data.fields,
    )

    if not result["success"]:
        return f"Ошибка поиска категорий: {result.get('error', 'Неизвестная ошибка')}"

    categories_res = await search_categories_by_collections(input_data.query_text, input_data.limit)

    if not categories_res:
        return "Категории не найдены."

    output = "Релевантные категории по коллекциям:\n\n"
    for collection_name, cat_list in categories_res.items():
        output += f"### Коллекция: {collection_name}\n"
        for cat in cat_list:
            output += f"- {cat.category_path} (score: {cat.score:.3f})\n"
        output += "\n"
    return output


async def get_category_hierarchy_tool(input_data: GetCategoryHierarchyInput) -> str:
    """Получение иерархии категорий с количеством чанков для указанных коллекций.
    Возвращает категории с количеством чанков, организованные по коллекциям.

    Полезно для:
    - Просмотра доступных категорий в коллекциях
    - Проверки структуры категорий
    - Получения количества документов в каждой категории
    """
    result = await _get_category_hierarchy_internal(
        collection_name=input_data.collection_name,
        depth=input_data.depth,
        categories=input_data.categories,
    )

    if result["success"]:
        results = result["data"].get("results", [])
        if not results:
            return "Иерархия категорий не найдена."

        output = "Иерархия категорий:\n\n"
        for coll in results:
            coll_name = coll.get("name", "unknown")
            categories_list = coll.get("categories", [])

            if not categories_list:
                output += f"### Коллекция: {coll_name}\n   (категории не найдены)\n\n"
            else:
                output += f"### Коллекция: {coll_name}\n"
                for cat in categories_list:
                    cat_path = cat.get("category_path", "unknown")
                    chunk_count = cat.get("chunk_count", 0)
                    output += f"- {cat_path} ({chunk_count} чанков)\n"
                output += "\n"
        return output
    else:
        return f"Ошибка получения иерархии категорий: {result['error']['message']}"


# --- Явный реестр MCP-инструментов ---
# Добавляйте новые инструменты сюда; магического discovery через vars() больше нет.
TOOLS = {
    "search_documents_tool": search_documents_tool,
    "search_categories_tool": search_categories_tool,
    "get_category_hierarchy_tool": get_category_hierarchy_tool,
}


# --- Простое кэширование информации о коллекциях (TTL 60 секунд) ---
_collections_cache: dict = {"data": None, "timestamp": 0}
_CACHE_TTL: int = 60


async def get_collections_info() -> str:
    """Возвращает строку с информацией о доступных коллекциях и их корневых категориях (с кэшированием через facet)."""
    now = time.time()
    if _collections_cache["data"] is not None and (now - _collections_cache["timestamp"]) < _CACHE_TTL:
        return _collections_cache["data"]

    try:
        client = get_client()
        collections_result = await client.get_collections()
        collections = collections_result.collections if hasattr(collections_result, 'collections') else []

        collection_info = []
        for collection in collections:
            if collection.name in ["categories"]:
                continue
            try:
                # Используем facet для получения уникальных category_level0
                # Поле должно быть ключевым (keyword) в Qdrant
                facet_result = await client.facet(
                    collection_name=collection.name,
                    key="category_level0",      # поле, где хранится корневая категория
                    limit=5                    # максимум уникальных значений
                )
                # facet_result - это объект с полями: hits (список значений и count)
                if hasattr(facet_result, 'hits'):
                    root_categories = [hit.value for hit in facet_result.hits if hit.value]
                else:
                    root_categories = []

                # Отображаем не более 5 категорий для краткости
                displayed = root_categories[:5]
                summary = ", ".join(displayed)
                if len(root_categories) > 5:
                    summary += f" и ещё {len(root_categories)-5}"
                collection_info.append(f"{collection.name} [{summary}]")
            except Exception as e:
                logger.error(f"Ошибка получения категорий для {collection.name}: {e}")

        if collection_info:
            result = "\nКоллекции [категории]:\n" + "\n".join(collection_info)
        else:
            result = "\nНет доступных коллекций."
    except Exception as e:
        logger.error(f"Ошибка подключения к Qdrant при получении списка коллекций: {e}")
        result = f"\nОшибка получения списка коллекций: {str(e)}"

    _collections_cache["data"] = result
    _collections_cache["timestamp"] = now
    return result
