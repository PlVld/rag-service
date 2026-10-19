"""Точный (лексический) поиск по тексту чанков.

Дополнительная ветка гибридного поиска: пользователь передаёт ключевые слова
или фразы (например, имена методов и команды), они ищутся по полнотекстовому
индексу `raw_text`, а результаты подмешиваются к семантическим и получают
приоритет.

Индекс `raw_text` создаётся скриптом `scripts/create_text_index.py`; новые
коллекции получают его автоматически. Если индекс недоступен, exact-ветка
не ломает поиск: коллекция пропускается с предупреждением в логе.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

MAX_EXACT_PHRASES = 10

# Qdrant WORD-токенизатор делит текст по не-алфавитно-цифровым символам;
# подчёркивание тоже разделитель. Такой же набор токенов используем в Python,
# чтобы пометка найденных фраз совпадала с фильтром Qdrant.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_phrases(phrases: Optional[Sequence[str]]) -> List[str]:
    """Очищает список фраз: trim, схлопывание пробелов, дедупликация."""
    if not phrases:
        return []

    result: List[str] = []
    seen = set()
    for phrase in phrases:
        if not isinstance(phrase, str):
            continue
        cleaned = " ".join(phrase.split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= MAX_EXACT_PHRASES:
            break
    return result


def tokenize(text: str) -> List[str]:
    """Токенизирует текст так же, как WORD-токенизатор Qdrant (без регистра)."""
    return _TOKEN_RE.findall((text or "").casefold())


def _contains_sequence(haystack: List[str], needle: List[str]) -> bool:
    """Проверяет вхождение последовательности токенов needle в haystack."""
    if not needle:
        return False
    size = len(needle)
    if size == 1:
        return needle[0] in haystack
    first = needle[0]
    for index in range(len(haystack) - size + 1):
        if haystack[index] == first and haystack[index:index + size] == needle:
            return True
    return False


def matched_phrases(text: str, phrases: Sequence[str]) -> List[str]:
    """Возвращает фразы, которые встречаются в тексте (без учёта регистра)."""
    tokens = tokenize(text)
    return [
        phrase for phrase in phrases
        if _contains_sequence(tokens, tokenize(phrase))
    ]


def _phrase_condition(phrase: str) -> qdrant_models.FieldCondition:
    """Условие Qdrant для одной фразы.

    Одно слово ищем через MatchText (совпадение токена), несколько слов —
    через MatchPhrase (последовательность токенов).
    """
    tokens = tokenize(phrase)
    if len(tokens) <= 1:
        match = qdrant_models.MatchText(text=phrase)
    else:
        match = qdrant_models.MatchPhrase(phrase=phrase)
    return qdrant_models.FieldCondition(key="raw_text", match=match)


def build_exact_filter(
    phrases: Sequence[str],
    phrase_mode: str,
    base_filter: Optional[qdrant_models.Filter] = None,
) -> qdrant_models.Filter:
    """Строит фильтр exact-ветки с учётом базового фильтра.

    phrase_mode="all" — должны встретиться все фразы;
    phrase_mode="any" — хотя бы одна.
    """
    conditions = [_phrase_condition(phrase) for phrase in phrases]
    base = [base_filter] if base_filter is not None else []

    if phrase_mode == "all":
        return qdrant_models.Filter(must=base + conditions)

    return qdrant_models.Filter(
        must=base,
        min_should=qdrant_models.MinShould(conditions=conditions, min_count=1),
    )


def exact_candidate_limit(limit: int) -> int:
    """Сколько exact-кандидатов запрашивать на коллекцию."""
    return min(max(limit * 4, 20), 200)


async def perform_exact_search(
    client,
    collection_name: str,
    query_vector: Sequence[float],
    base_filter: Optional[qdrant_models.Filter],
    phrases: Sequence[str],
    phrase_mode: str,
    limit: int,
) -> List[qdrant_models.ScoredPoint]:
    """Ищет exact-совпадения и ранжирует их косинусным сходством.

    Фильтр гарантирует лексическое совпадение, а векторный запрос — порядок
    кандидатов: лучшие по смыслу среди точных совпадений идут первыми.
    """
    exact_filter = build_exact_filter(phrases, phrase_mode, base_filter)
    try:
        response = await client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=exact_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
    except UnexpectedResponse as exc:
        # Например, на коллекции ещё нет полнотекстового индекса `raw_text`.
        # Семантическая ветка при этом должна продолжить работать.
        logger.warning(
            "Exact search in collection '%s' failed: %s", collection_name, exc
        )
        return []

    points: List[qdrant_models.ScoredPoint] = []
    for point in response.points:
        payload = dict(point.payload or {})
        text = payload.get("raw_text", "")
        matched = matched_phrases(text, phrases)
        if not matched:
            # Расхождение токенизации Qdrant и Python маловероятно, но если
            # оно случилось — не теряем совпадение, помечаем все фразы.
            matched = list(phrases)

        payload["_collection_name"] = collection_name
        payload["_exact_match"] = True
        payload["_matched_phrases"] = matched
        point.payload = payload
        points.append(point)

    return points


def _point_key(point: qdrant_models.ScoredPoint) -> tuple:
    payload = point.payload or {}
    return (payload.get("_collection_name", ""), str(point.id))


def merge_exact_points(
    semantic_points: Sequence[qdrant_models.ScoredPoint],
    exact_points: Sequence[qdrant_models.ScoredPoint],
) -> List[qdrant_models.ScoredPoint]:
    """Объединяет ветки, помечая exact-дубликаты и не теряя score семантики."""
    merged: Dict[tuple, qdrant_models.ScoredPoint] = {}
    order: List[tuple] = []

    for point in semantic_points:
        key = _point_key(point)
        if key not in merged:
            order.append(key)
        merged[key] = point

    for point in exact_points:
        key = _point_key(point)
        existing = merged.get(key)
        if existing is None:
            merged[key] = point
            order.append(key)
            continue

        payload = existing.payload or {}
        payload["_exact_match"] = True
        payload["_matched_phrases"] = sorted(
            set(payload.get("_matched_phrases") or [])
            | set((point.payload or {}).get("_matched_phrases") or [])
        )
        existing.payload = payload

    return [merged[key] for key in order]


def is_exact_point(point: qdrant_models.ScoredPoint) -> bool:
    return bool((point.payload or {}).get("_exact_match"))


def hybrid_sort_key(point: qdrant_models.ScoredPoint) -> tuple:
    """Ключ сортировки: exact-совпадения первыми, затем по числу фраз и score."""
    payload = point.payload or {}
    return (
        0 if payload.get("_exact_match") else 1,
        -len(payload.get("_matched_phrases") or []),
        -float(point.score or 0.0),
    )


def sort_hybrid_points(
    points: Sequence[qdrant_models.ScoredPoint],
) -> List[qdrant_models.ScoredPoint]:
    return sorted(points, key=hybrid_sort_key)


def filter_by_min_score(
    points: Sequence[qdrant_models.ScoredPoint],
    min_score: Optional[float],
) -> List[qdrant_models.ScoredPoint]:
    """Порог применяется только к семантическим результатам.

    Exact-совпадение важно само по себе: команда или редкий идентификатор
    может иметь низкое косинусное сходство, но быть единственным точным
    ответом.
    """
    if min_score is None:
        return list(points)
    return [p for p in points if is_exact_point(p) or p.score >= min_score]
