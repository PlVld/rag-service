# Contributing Guide

Спасибо за интерес к участию в проекте! Этот документ содержит правила и рекомендации для контрибьюторов.

## 📋 Содержание

- [Code of Conduct](#code-of-conduct)
- [Как начать](#как-начать)
- [Process](#process)
- [Style Guide](#style-guide)
- [Commit Messages](#commit-messages)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Requests](#pull-requests)

---

## Code of Conduct

Этот проект придерживается стандартов открытого сообщества. Ожидайте уважительного отношения от всех участников. Неприемлемое поведение (толерантность, оскорбления, личная агрессия) не допускается.

## Как начать

### 1. Fork и клонирование

```bash
# Создайте fork через GitHub UI
# Затем клонируйте:
git clone https://github.com/YOUR_USERNAME/rag-service.git
cd rag-service

# Добавьте upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/rag-service.git
```

### 2. Настройка окружения

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Установка зависимостей
pip install -r requirements.txt
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Установка dev-зависимостей
pip install pytest pytest-cov black flake8 mypy

# Запуск Qdrant для разработки
docker-compose up -d qdrant
```

### 3. Создание ветки

```bash
git checkout -b feature/your-feature-name
# или
git checkout -b fix/your-bug-fix
```

## Process

### Workflow

```
1. Выберите issue или создайте новый
2. Опишите ваш подход в комментариях
3. Создайте branch от main
4. Внесите изменения
5. Напишите/обновите тесты
6. Обновите документацию
7. Создайте Pull Request
```

### Type conventions

| Префикс | Назначение |
|---------|-----------|
| `feat:` | Новая функциональность |
| `fix:` | Исправление бага |
| `docs:` | Изменения в документации |
| `style:` | Форматирование, точки с запятой и т.д. |
| `refactor:` | Рефакторинг кода |
| `test:` | Добавление или обновление тестов |
| `chore:` | Обновление зависимостей, конфигурации |
| `perf:` | Оптимизация производительности |

## Style Guide

### Python Code Style

- **Black** — форматирование кода (line length: 88)
- **PEP 8** — основные конвенции
- **Type hints** — обязательны для публичных функций

```python
# ✅ Хорошо
async def search_documents(
    query: str,
    limit: int = 10,
    collection_name: Optional[str] = None
) -> List[DocumentResult]:
    """Search documents by semantic similarity.
    
    Args:
        query: Search query text
        limit: Maximum number of results
        collection_name: Optional collection name
        
    Returns:
        List of matching documents
    """
    ...

# ❌ Плохо
def search(query, limit=10): # no type hints, no docstring
    ...
```

### Naming Conventions

- `snake_case` для функций и переменных
- `PascalCase` для классов
- `UPPER_SNAKE_CASE` для констант
- `_private` для внутренних методов

### File Structure

```
app/
├── api/          # API endpoints
├── core/         # Core functionality
├── models/       # Pydantic models
├── repository/   # Data access
├── text_cleaning/ # Text processing
tests/
├── conftest.py   # Shared fixtures
└── test_*.py     # Test files
```

## Commit Messages

Используйте [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Формат: type(scope): description

git commit -m "feat(api): add grouped search endpoint"
git commit -m "fix(search): fix category path normalization"
git commit -m "docs(readme): add quick start section"
git commit -m "refactor(chunking): simplify chunk factory logic"
```

### Правила

- **Начинать с маленькой буквы** после двоеточия
- **Не более 72 символов** в первой строке
- **Использовать императив** ("add"而不是"added")
- **Не завершать** точкой в первой строке

## Testing

### Запуск тестов

```bash
# Все тесты
pytest

# С покрытием
pytest --cov=app --cov-report=html

# Конкретный файл
pytest tests/test_rest_api_documents.py -v

# Быстрые тесты
pytest -m "not integration"
```

### Правила тестирования

- **Coverage** ≥ 80% для нового кода
- **Unit тесты** для бизнес-логики
- **Integration тесты** для API эндпоинтов
- **Mock Qdrant** в unit тестах

```python
# ✅ Хорошо
@pytest.mark.asyncio
async def test_search_returns_results(mock_qdrant_client):
    # Arrange
    query = "тестовый запрос"
    
    # Act
    response = await search_documents(query)
    
    # Assert
    assert response.success is True
    assert len(response.data["results"]) > 0

# ❌ Плохо
def test_search():
    # нет тестов, нет ассертов
    pass
```

## Documentation

### Обязательная документация

При изменении кода обновляйте:

1. **README.md** — если изменился quick start или ключевые возможности
2. **docs/api/README.md** — если изменились API эндпоинты
3. **CHANGELOG.md** — запишите изменения в раздел `[Unreleased]`
4. **Docstrings** — для всех публичных функций и классов

### Docstring Style (Google Style)

```python
def process_documents(
    request: DocumentsUploadRequest,
    batch_writer: QdrantBatchWriter,
) -> Tuple[List[str], List[str]]:
    """Process documents for vector database indexing.
    
    Args:
        request: Upload request with document data
        batch_writer: Batch writer for Qdrant operations
        
    Returns:
        Tuple of (updated_source_ids, created_point_ids)
        
    Raises:
        ValidationError: If document data is invalid
        QdrantError: If Qdrant operations fail
    """
```

## Pull Requests

### Чеклист перед PR

- [ ] Код следует style guide
- [ ] Добавлены/обновлены тесты
- [ ] Пройдут все CI проверки
- [ ] Обновлена документация
- [ ] CHANGELOG.md обновлён
- [ ] Commits следуют Conventional Commits
- [ ] Актуально с main веткой upstream

### Описание PR

```markdown
## Description

Краткое описание изменений

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

Опишите как тестировали изменения

## Checklist

- [ ] Мой код следует style guide проекта
- [ ] Я выполнил self-review
- [ ] Я обновил документацию
- [ ] Я добавил тесты
- [ ] Все тесты прошли успешно
```

### Процесс ревью

1. Maintainer проводит code review
2. Возможно запрос на изменения
3. После одобрения PR merges в main
4. Automatic CI checks должны пройти

## Контакты

- GitHub Issues: для багов и фич-реквестов
- GitHub Discussions: для общих вопросов
- Email: [your@email.com](mailto:your@email.com)

---

Спасибо за вклад в проект! 🎉
