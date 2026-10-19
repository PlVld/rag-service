"""Создание полнотекстового индекса на raw_text.

Индекс нужен для точного (лексического) поиска по фразам в гибридном режиме:
параметр `exact_phrases` в поиске документов. Без индекса Qdrant выполняет
регистрозависимый перебор подстрокой, а `MatchPhrase` не работает.

По умолчанию скрипт работает в режиме dry-run и ничего не меняет.
Создание индекса — только с флагом --apply.

Примеры:
    # показать, где индекса нет
    python scripts/create_text_index.py

    # создать индекс в одной коллекции и дождаться завершения
    python scripts/create_text_index.py --collection 1c_integration --apply

    # создать во всех коллекциях документов
    python scripts/create_text_index.py --apply
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional

# Запуск как `python scripts/create_text_index.py`: добавляем корень проекта
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client.http import models as qdrant_models  # noqa: E402

from app.api.health import get_client  # noqa: E402
from app.core.config import settings  # noqa: E402


FIELD_NAME = "raw_text"


def text_index_params() -> qdrant_models.TextIndexParams:
    """Параметры индекса, согласованные с exact-поиском."""
    return qdrant_models.TextIndexParams(
        type=qdrant_models.TextIndexType.TEXT,
        tokenizer=qdrant_models.TokenizerType.WORD,
        lowercase=True,
        phrase_matching=True,
        min_token_len=1,
        max_token_len=40,
    )


async def list_collections(client) -> List[str]:
    response = await client.get_collections()
    return [
        c.name for c in response.collections
        if c.name != settings.category_collection
    ]


async def index_state(client, collection_name: str) -> Optional[str]:
    """Возвращает тип существующего индекса raw_text или None."""
    info = await client.get_collection(collection_name)
    schema = info.payload_schema or {}
    existing = schema.get(FIELD_NAME)
    if existing is None:
        return None
    data_type = getattr(existing, "data_type", None)
    return str(data_type) if data_type is not None else "unknown"


async def process_collection(client, collection_name: str, apply: bool, wait: bool) -> None:
    info = await client.get_collection(collection_name)
    points_count = info.points_count or 0
    existing = await index_state(client, collection_name)

    if existing:
        print(f"  {collection_name}: индекс уже есть ({existing}), пропуск")
        return

    print(f"  {collection_name}: точек {points_count}, индекса нет")
    if not apply:
        print("    dry-run: индекс не создаётся (добавьте --apply)")
        return

    started = time.monotonic()
    await client.create_payload_index(
        collection_name=collection_name,
        field_name=FIELD_NAME,
        field_schema=text_index_params(),
        wait=wait,
    )
    elapsed = time.monotonic() - started
    print(f"    индекс создан за {elapsed:.2f} c (wait={wait})")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Создать полнотекстовый индекс raw_text для exact-поиска"
    )
    parser.add_argument(
        "--collection", action="append", dest="collections",
        help="Коллекция (можно несколько раз). По умолчанию все, кроме служебной.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Создать индекс. Без флага — только показать состояние (dry-run).",
    )
    parser.add_argument(
        "--no-wait", action="store_false", dest="wait",
        help="Не ждать завершения построения индекса.",
    )
    parser.set_defaults(wait=True)
    args = parser.parse_args()

    client = get_client()
    try:
        collections = args.collections or await list_collections(client)
        if not collections:
            print("Коллекции не найдены.")
            return 0

        mode = "APPLY" if args.apply else "DRY-RUN"
        print(f"Режим: {mode}. Коллекций: {len(collections)}")
        for name in collections:
            try:
                await process_collection(client, name, args.apply, args.wait)
            except Exception as e:  # noqa: BLE001 - отчёт по каждой коллекции
                print(f"  {name}: ОШИБКА {type(e).__name__}: {e}")
                return 1
    finally:
        await client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
