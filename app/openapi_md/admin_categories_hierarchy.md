Возвращает категории с количеством чанков для указанной коллекции.

**Параметры:**
- `collection_name`: имя коллекции (опционально, если не указан - возвращает категории для всех коллекций)
- `depth`: количество уровней для получения (0, 1-10)
  - `0` - только список коллекций без категорий
  - `1-10` - иерархия категорий (максимум 10 уровней)
- `categories`: массив полных путей категорий (опционально)
  - Разделитель уровней - ` / ` (пробел-слэш-пробел), также принимается `/` без пробелов
  - Если указаны - возвращает только коллекции, содержащие эти категории
  - Если не указаны - возвращает все категории для указанных/всех коллекций
- `limit`: максимальное суммарное количество категорий в ответе (по умолчанию 5000)
  - Коллекции обходятся по порядку, пока не исчерпан лимит
  - `null` - без ограничения
  - При обрезке `truncated=true`, в каждой коллекции заполняются
    `total_categories` и `truncated`

**Логика:**
- Если `categories` не указаны и `depth=0` - возвращает список коллекций без категорий
- Если `categories` не указаны и `depth>0` - возвращает первые `depth` уровней
- Если `categories` указаны - для каждого пути возвращает `depth` вложенных уровней
- Если указана `collection_name` и она не существует - возвращает ошибку 404

**Пример ответа (depth=1, categories не указаны):**
```json
{
  "results": [
    {"name": "collection1", "total_categories": 2, "truncated": false, "categories": [
      {"category_path": "Docs", "chunk_count": 150},
      {"category_path": "API", "chunk_count": 50}
    ]}
  ],
  "limit": 5000,
  "truncated": false,
  "total_categories": 2,
  "shown_categories": 2
}
```

**Пример ответа при обрезке (limit=1):**
```json
{
  "results": [
    {"name": "collection1", "total_categories": 2, "truncated": true, "categories": [
      {"category_path": "API", "chunk_count": 50}
    ]},
    {"name": "collection2", "total_categories": 3, "truncated": true, "categories": []}
  ],
  "limit": 1,
  "truncated": true,
  "total_categories": 5,
  "shown_categories": 1
}
```

**Пример ответа (depth=0):**
```json
{
  "results": [
    {"name": "collection1", "total_categories": 0, "truncated": false, "categories": []},
    {"name": "collection2", "total_categories": 0, "truncated": false, "categories": []}
  ],
  "limit": 5000,
  "truncated": false,
  "total_categories": 0,
  "shown_categories": 0
}
```

**Пример ответа (categories указанны, несуществующая категория):**
```json
{
  "results": []
}
```

**Ошибки:**
- 404 - коллекция не найдена (если указана `collection_name`)
- 422 - неверные параметры (например, `depth > 10`)
