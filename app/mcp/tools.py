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
    GetGroupedTextInput,
    _search_documents_internal,
    _search_categories_internal,
    _get_category_hierarchy_internal,
    _get_grouped_text_internal,
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

    Для каждого результата выводятся коллекция (`collection_name`), категория
    (`category_path`), идентификатор документа (`source_id`), номера чанков,
    из которых собран текст (диапазон для group=true, один номер для
    group=false), и общее количество чанков в документе.

    Параметр `min_score` (0..1) отбрасывает результаты с косинусным сходством
    ниже указанного (по умолчанию 0.55); `null` — тоже значение по умолчанию,
    `0` — без ограничения.

    Параметр `exact_phrases` добавляет точный (лексический) поиск по ключевым
    словам и фразам, например по именам методов и командам. Найденные
    совпадения идут первыми и не отбрасываются параметром `min_score`.
    `phrase_mode` управляет режимом: `any` — достаточно любой фразы,
    `all` — должны встретиться все.
    """
    result = await _search_documents_internal(
        query_text=input_data.query_text,
        collection_name=input_data.collection_name,
        limit=input_data.limit,
        filter_criteria=input_data.filter_criteria,
        include_old_versions=input_data.include_old_versions,
        max_text_length=input_data.max_text_length,
        group=input_data.group,
        min_score=input_data.min_score,
        exact_phrases=input_data.exact_phrases,
        phrase_mode=input_data.phrase_mode,
    )

    if result["success"]:
        results = result["data"]["results"]
        if not results:
            return "По вашему запросу ничего не найдено."

        output = f"Найдено {len(results)} результатов:\n\n"
        for i, r in enumerate(results, 1):
            # Используем подготовленный snippet из mcp_server (уже с учетом max_text_length)
            snippet = r["document"]
            output += f"{i}. [score={r['score']:.3f}]{_format_chunk_meta(r)} {snippet}\n\n"
        return output
    else:
        return f"Ошибка поиска: {result['error']['message']}"


def _format_chunk_meta(result: dict) -> str:
    """Формирует метаданные результата: коллекция, категория, id документа,
    номера чанков и всего чанков."""
    parts = []

    collection_name = result.get("collection_name")
    if collection_name:
        parts.append(f"collection_name={collection_name}")

    category_path = result.get("category_path")
    if category_path:
        if isinstance(category_path, list):
            category_path = " / ".join(str(part) for part in category_path)
        parts.append(f"category_path={category_path}")

    source_id = result.get("source_id")
    if source_id:
        parts.append(f"source_id={source_id}")

    if result.get("exact_match"):
        parts.append("exact_match=true")
    matched_phrases = result.get("matched_phrases")
    if matched_phrases:
        parts.append(f"matched_phrases={', '.join(str(p) for p in matched_phrases)}")

    total_chunks = result.get("total_chunks")
    first_idx = result.get("first_chunk_index")
    last_idx = result.get("last_chunk_index")
    chunk_idx = result.get("chunk_index")

    if first_idx is not None and last_idx is not None:
        chunks_label = f"чанк {first_idx}" if first_idx == last_idx else f"чанки {first_idx}–{last_idx}"
    elif chunk_idx is not None:
        chunks_label = f"чанк {chunk_idx}"
    else:
        chunks_label = None

    if chunks_label:
        if total_chunks is not None:
            chunks_label += f" из {total_chunks}"
        parts.append(chunks_label)
    elif total_chunks is not None:
        parts.append(f"всего чанков: {total_chunks}")

    return f" ({', '.join(parts)})" if parts else ""


async def search_categories_tool(input_data: SearchCategoriesInput) -> str:
    """Поиск категорий документов, соответствующих текстовому запросу.
    Категории возвращаются сгруппированными по коллекциям.

    Параметр `min_score` (0..1) отбрасывает категории с косинусным сходством
    ниже указанного (по умолчанию 0.55); `null` — тоже значение по умолчанию,
    `0` — без ограничения.

    Параметр `exact_phrases` добавляет точный поиск по путям категорий;
    найденные совпадения идут первыми и не отбрасываются параметром `min_score`.
    """
    from app.api.categories import search_categories_by_collections

    result = await _search_categories_internal(
        query_text=input_data.query_text,
        limit=input_data.limit,
        fields=input_data.fields,
        min_score=input_data.min_score,
        exact_phrases=input_data.exact_phrases,
        phrase_mode=input_data.phrase_mode,
    )

    if not result["success"]:
        return f"Ошибка поиска категорий: {result.get('error', 'Неизвестная ошибка')}"

    categories_res = await search_categories_by_collections(
        input_data.query_text, input_data.limit, min_score=input_data.min_score,
        exact_phrases=input_data.exact_phrases, phrase_mode=input_data.phrase_mode,
    )

    if not categories_res:
        return "Категории не найдены."

    output = "Релевантные категории по коллекциям:\n\n"
    for collection_name, cat_list in categories_res.items():
        output += f"### Коллекция: {collection_name}\n"
        for cat in cat_list:
            exact_label = " [exact]" if cat.exact_match else ""
            output += f"- {cat.category_path} (score: {cat.score:.3f}){exact_label}\n"
        output += "\n"
    return output


async def get_category_hierarchy_tool(input_data: GetCategoryHierarchyInput) -> str:
    """Получение иерархии категорий с количеством чанков для указанных коллекций.
    Возвращает категории с количеством чанков, организованные по коллекциям.

    Полезно для:
    - Просмотра доступных категорий в коллекциях
    - Проверки структуры категорий
    - Получения количества документов в каждой категории

    Параметр `limit` (по умолчанию 5000) ограничивает суммарное количество
    категорий в ответе. Если категорий больше, вывод обрезается, а в конце
    указывается, сколько показано из всего. Чтобы получить больше, увеличьте
    `limit`, укажите `collection_name` или сузьте `categories`.
    """
    result = await _get_category_hierarchy_internal(
        collection_name=input_data.collection_name,
        depth=input_data.depth,
        categories=input_data.categories,
        limit=input_data.limit,
    )

    if result["success"]:
        data = result["data"]
        results = data.get("results", [])
        if not results:
            return "Иерархия категорий не найдена."

        output = "Иерархия категорий:\n\n"
        for coll in results:
            coll_name = coll.get("name", "unknown")
            categories_list = coll.get("categories", [])
            total = coll.get("total_categories", len(categories_list))
            truncated = coll.get("truncated", False)

            if not categories_list:
                if truncated and total:
                    output += (
                        f"### Коллекция: {coll_name}\n"
                        f"   (категории не показаны из-за лимита: всего {total})\n\n"
                    )
                else:
                    output += f"### Коллекция: {coll_name}\n   (категории не найдены)\n\n"
            else:
                output += f"### Коллекция: {coll_name}\n"
                for cat in categories_list:
                    cat_path = cat.get("category_path", "unknown")
                    chunk_count = cat.get("chunk_count", 0)
                    output += f"- {cat_path} ({chunk_count} чанков)\n"
                if truncated and total > len(categories_list):
                    output += f"   … показано {len(categories_list)} из {total} категорий\n"
                output += "\n"

        if data.get("truncated"):
            output += (
                f"Показано {data.get('shown_categories', 0)} из "
                f"{data.get('total_categories', 0)} категорий "
                f"(limit={data.get('limit')}). Увеличьте limit, укажите "
                "collection_name или сузьте categories.\n"
            )
        return output
    else:
        return f"Ошибка получения иерархии категорий: {result['error']['message']}"


async def get_grouped_text_tool(input_data: GetGroupedTextInput) -> str:
    """Возвращает сгруппированный текст чанков без семантического поиска.

    Чанки отбираются по коллекции, категориям (включая вложенные) и списку
    документов (source_id), затем группируются по паре «документ + категория».
    Внутри группы текст склеивается в порядке chunk_index.

    Для каждой группы выводятся идентификатор документа (`source_id`),
    коллекция, категория, номера чанков и общее количество чанков в документе.
    """
    result = await _get_grouped_text_internal(
        collection_name=input_data.collection_name,
        categories=input_data.categories,
        documents=input_data.documents,
        include_old_versions=input_data.include_old_versions,
        max_text_length=input_data.max_text_length,
        limit=input_data.limit,
    )

    if not result["success"]:
        return f"Ошибка получения сгруппированного текста: {result['error']['message']}"

    results = result["data"]["results"]
    if not results:
        return "Группы не найдены."

    output = f"Найдено {len(results)} групп:\n\n"
    for i, r in enumerate(results, 1):
        output += f"{i}.{_format_grouped_meta(r)} {r['document']}\n\n"
    return output


def _format_chunk_numbers(indices: list, total_chunks) -> str:
    """Сжимает номера чанков в диапазоны: 1–3, 5, 7–8."""
    if not indices:
        return f"всего чанков: {total_chunks}" if total_chunks is not None else ""

    ranges = []
    start = prev = indices[0]
    for index in indices[1:]:
        if index == prev + 1:
            prev = index
        else:
            ranges.append((start, prev))
            start = prev = index
    ranges.append((start, prev))

    labels = [str(a) if a == b else f"{a}–{b}" for a, b in ranges]
    label = ("чанк " if len(indices) == 1 else "чанки ") + ", ".join(labels)
    if total_chunks is not None:
        label += f" из {total_chunks}"
    return label


def _format_grouped_meta(result: dict) -> str:
    """Формирует метаданные группы для вывода get_grouped_text_tool."""
    parts = []
    if result.get("source_id"):
        parts.append(f"source_id={result['source_id']}")
    if result.get("collection_name"):
        parts.append(f"collection_name={result['collection_name']}")
    if result.get("category_path"):
        parts.append(f"category_path={result['category_path']}")

    chunks_label = _format_chunk_numbers(
        result.get("chunk_indices") or [],
        result.get("total_chunks"),
    )
    if chunks_label:
        parts.append(chunks_label)

    return f" ({', '.join(parts)})" if parts else ""


# --- Явный реестр MCP-инструментов ---
# Добавляйте новые инструменты сюда; магического discovery через vars() больше нет.
TOOLS = {
    "search_documents_tool": search_documents_tool,
    "search_categories_tool": search_categories_tool,
    "get_category_hierarchy_tool": get_category_hierarchy_tool,
    "get_grouped_text_tool": get_grouped_text_tool,
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
