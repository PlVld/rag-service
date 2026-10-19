Актуализирует коллекцию `categories` по путям категорий указанной коллекции
и чинит `category_id_levelN` в чанках коллекции-источника.

Используется после переноса коллекций с другой машины: пути категорий в
чанках те же, но `category_id_levelN` могут указывать на ID узлов прежней
коллекции `categories`. Общая коллекция `categories` при этом может быть
пустой или устаревшей.

**Параметры:**
- `collection_name`: имя коллекции-источника. Не может совпадать с
  `CATEGORY_COLLECTION` (по умолчанию `categories`)
- `include_old_versions`: учитывать точки с `is_latest=false`
  (по умолчанию `false` — только актуальные версии)
- `dry_run`: только посчитать изменения, не считать эмбеддинги и не писать
  в Qdrant (по умолчанию `false`)

**Что делает:**
1. Читает уникальные пути категорий из коллекции-источника полным `scroll`.
   Уровни берутся из `category_level0..N`, а не разбираются из
   `category_path`: название уровня может само содержать ` / `
   (например, `Как работает ЕСТЬNULL (ISNULL / COALESCE)`).
   `category_path` используется только как fallback, если уровней нет.
2. Разворачивает путь `A / B / C` во все узлы: `A`, `A / B`, `A / B / C`.
3. Для каждого узла вычисляет детерминированный UUID:
   `uuid5(NAMESPACE_DNS, category_path)` и ожидаемый payload
   (`category_name`, `category_path`, `category_level`, `parent_id`,
   `category_id`, `category_levelN`, `category_id_levelN`).
4. Читает существующие узлы из `categories` батчами и сравнивает payload.
5. Создаёт отсутствующие узлы и исправляет узлы с неверным или устаревшим
   payload. Эмбеддинги считаются только для изменённых узлов.
6. Если коллекции `categories` нет, создаёт её со штатными payload-индексами.
7. Приводит `category_id_levelN` в чанках коллекции-источника к
   детерминированным ID узлов для их пути (тем же, что вычислены в п. 3).
   Если в чанке есть поле `category_id`, оно тоже обновляется до ID листового
   узла. Точки без `category_levelN` не трогаются: путь из `category_path`
   неоднозначен, если название уровня содержит ` / `.

**Идемпотентность:** повторный запуск на неизменной коллекции ничего не
пишет, не считает эмбеддинги и возвращает `created=0`, `updated=0`,
`points_updated=0`.

**Ограничения:**
- Существующие категории не удаляются: коллекция `categories` общая для
  нескольких коллекций документов.
- В коллекции-источнике меняются только поля `category_id_levelN` и, если
  он есть, `category_id`. Текст, векторы и остальной payload не затрагиваются.
- Уровни читаются из `category_levelN`; `category_path` используется только
  как fallback, если уровней в payload нет.

**Пример запроса:**
```json
{
  "collection_name": "1c_integration",
  "include_old_versions": false,
  "dry_run": true
}
```

**Пример ответа:**
```json
{
  "collection_name": "1c_integration",
  "dry_run": true,
  "include_old_versions": false,
  "scan_mode": "scroll",
  "points_scanned": 27273,
  "source_category_paths": 314,
  "category_nodes": 402,
  "parent_nodes": 88,
  "missing": 12,
  "outdated": 3,
  "unchanged": 387,
  "created": 0,
  "updated": 0,
  "embedded": 0,
  "points_with_wrong_category_ids": 1024,
  "points_updated": 0,
  "duration_ms": 145,
  "sample_missing": ["Интеграция / API / Методы / Новый метод"],
  "sample_outdated": ["Интеграция / API / Методы"],
  "sample_wrong_category_ids": ["Интеграция / API / Методы (512 точек)"]
}
```

При `dry_run=true` поля `points_with_wrong_category_ids` и
`sample_wrong_category_ids` показывают масштаб расхождения, а `points_updated`
остаётся нулём.

**Ошибки:**
- `400` — `collection_name` совпадает с коллекцией категорий
- `404` — коллекция-источник не найдена
- `500` — ошибка Qdrant или эмбеддингов
