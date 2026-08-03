import logging
import time
from typing import Dict, List, Set, Tuple
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from app.api.health import get_client
from app.core.embeddings import get_embedding_dimension
from app.core.config import settings

logger = logging.getLogger(__name__)

# Максимальное количество точек в одном batch_update_points запросе к Qdrant.
# Слишком большие батчи приводят к обрыву соединения (WinError 10053)
# из-за превышения тайм-аута или лимита размера HTTP-запроса.
BATCH_UPSERT_SIZE = settings.upload_batch_size


class QdrantBatchWriter:
    """
    Сборщик операций для атомарного выполнения минимальным числом запросов.
    Использует batch_update_points для выполнения всех операций в одном запросе.
    """

    def __init__(self):
        self._points_by_collection: Dict[str, List[qdrant_models.PointStruct]] = {}
        self._mark_not_latest: Dict[str, List[Tuple[str, int]]] = {}  # (source_id, keep_version)
        self._collections_checked: Set[str] = set()

    def add_point(self, collection_name: str, point: qdrant_models.PointStruct):
        self._points_by_collection.setdefault(collection_name, []).append(point)

    def mark_old_versions_not_latest(self, collection_name: str, source_id: str, keep_version: int):
        """Помечает все версии с данным source_id, кроме keep_version, как is_latest=False."""
        self._mark_not_latest.setdefault(collection_name, []).append((source_id, keep_version))

    @staticmethod
    async def _ensure_collection(collection_name: str):
        client = get_client()
        try:
            client.get_collection(collection_name)
        except UnexpectedResponse as e:
            if "Not found: Collection" in str(e):
                logger.info(f"Creating collection {collection_name}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=get_embedding_dimension(),
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
                
                # Создаем индексы для часто используемых полей
                try:
                    # Индекс для source_id - используется для поиска и версионности
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="source_id",
                        field_type=qdrant_models.PayloadSchemaType.KEYWORD
                    )

                    # Индекс для is_latest - используется для фильтрации актуальных версий
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="is_latest",
                        field_type=qdrant_models.PayloadSchemaType.BOOL
                    )

                    # Индекс для version - используется для версионности
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="version",
                        field_type=qdrant_models.PayloadSchemaType.INTEGER
                    )

                    # Индекс для category_id - используется для поиска по категориям
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="category_id",
                        field_type=qdrant_models.PayloadSchemaType.KEYWORD
                    )

                    # Индекс для category_path_ids - используется для RRF-поиска по категориям
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="category_path_ids",
                        field_type=qdrant_models.PayloadSchemaType.KEYWORD
                    )

                    # Индекс для parent_id - используется для иерархии категорий
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="parent_id",
                        field_type=qdrant_models.PayloadSchemaType.KEYWORD
                    )

                    # Индекс для category_path - используется для фильтрации по пути
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="category_path",
                        field_type=qdrant_models.PayloadSchemaType.TEXT
                    )

                    # Индексы для уровней категорий category_level0..9 - фильтрация по имени категории
                    for level in range(10):
                        client.create_payload_index(
                            collection_name=collection_name,
                            field_name=f"category_level{level}",
                            field_type=qdrant_models.PayloadSchemaType.KEYWORD
                        )

                    # Индексы для ID уровней категорий category_id_level0..9 - фильтрация по ID категории
                    for level in range(10):
                        client.create_payload_index(
                            collection_name=collection_name,
                            field_name=f"category_id_level{level}",
                            field_type=qdrant_models.PayloadSchemaType.KEYWORD
                        )

                    logger.info(f"Created payload indexes for collection {collection_name}")
                except Exception as e:
                    logger.warning(f"Failed to create payload indexes for {collection_name}: {e}")
            else:
                raise

    async def commit(self):
        start_time = time.time()
        client = get_client()

        # 1. Создать недостающие коллекции
        for collection_name in self._points_by_collection.keys():
            if collection_name not in self._collections_checked:
                await self._ensure_collection(collection_name)
                self._collections_checked.add(collection_name)

        # 2. Для каждой коллекции формируем batch-операции
        for collection_name, points in self._points_by_collection.items():
            operations = []
            markers = self._mark_not_latest.get(collection_name, [])

            # Операция 1: вставка новых точек
            if points:
                operations.append(
                    qdrant_models.UpsertOperation(
                        upsert=qdrant_models.PointsList(points=points)
                    )
                )

            # Операция 2: пометка старых версий (исключая keep_version)
            if markers:
                # Строим фильтр: source_id in [source_ids] AND is_latest=True AND version != keep_version
                # Для каждого source_id свой keep_version
                # Qdrant поддерживает сложные условия, но проще сделать отдельный set_payload для каждого source_id
                for source_id, keep_version in markers:
                    filter_cond = qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="source_id",
                                match=qdrant_models.MatchValue(value=source_id)
                            ),
                            qdrant_models.FieldCondition(
                                key="is_latest",
                                match=qdrant_models.MatchValue(value=True)
                            ),
                        ],
                        must_not=[
                            qdrant_models.FieldCondition(
                                key="version",
                                match=qdrant_models.MatchValue(value=keep_version)
                            ),
                        ]
                    )
                    operations.append(
                        qdrant_models.SetPayloadOperation(
                            set_payload=qdrant_models.SetPayload(
                                payload={"is_latest": False},
                                filter=filter_cond,
                            )
                        )
                    )

            # Если есть операции, выполняем батчами
            if operations:
                try:
                    write_start = time.time()

                    # Разбиваем upsert на чанки по BATCH_UPSERT_SIZE точек,
                    # чтобы избежать обрыва соединения при большом количестве точек.
                    # Остальные операции (set_payload) отправляются в финальном батче.
                    upsert_ops = [op for op in operations if isinstance(op, qdrant_models.UpsertOperation)]
                    other_ops = [op for op in operations if not isinstance(op, qdrant_models.UpsertOperation)]

                    batch_count = 0
                    for upsert_op in upsert_ops:
                        all_points = upsert_op.upsert.points
                        for i in range(0, len(all_points), BATCH_UPSERT_SIZE):
                            chunk = all_points[i:i + BATCH_UPSERT_SIZE]
                            chunk_ops = [
                                qdrant_models.UpsertOperation(
                                    upsert=qdrant_models.PointsList(points=chunk)
                                )
                            ]
                            client.batch_update_points(
                                collection_name=collection_name,
                                update_operations=chunk_ops,
                                wait=True
                            )
                            batch_count += 1
                            logger.debug(
                                f"Upserted batch {batch_count}: {len(chunk)} points "
                                f"to {collection_name}"
                            )

                    # Отправляем остальные операции (mark_not_latest и т.д.)
                    if other_ops:
                        client.batch_update_points(
                            collection_name=collection_name,
                            update_operations=other_ops,
                            wait=True
                        )

                    write_elapsed = time.time() - write_start
                    logger.info(
                        f"Batch update success for {collection_name}: "
                        f"upserted {len(points) if points else 0} points "
                        f"in {batch_count} batch(es), "
                        f"marked {len(markers)} source_ids (excluding keep_version), "
                        f"write_time={write_elapsed:.3f}s"
                    )
                except Exception:
                    logger.exception(f"Batch update failed for {collection_name}")
                    raise
        
        total_elapsed = time.time() - start_time
        logger.info(f"Total batch commit time: {total_elapsed:.3f}s")