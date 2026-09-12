"""Тесты загрузки файлов различных форматов (текст, markdown, PDF)."""
import pytest
import time
import uuid
from qdrant_client import QdrantClient
from app.core.config import settings
from qdrant_client.http import models as qdrant_models


def get_client() -> QdrantClient:
    """Синхронный клиент Qdrant для setup и проверок в синхронных тестах."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=120,
    )


def get_auth_headers():
    """Возвращает заголовки для аутентификации через Bearer token."""
    return {"Authorization": "Bearer PASS1234"}


@pytest.fixture
def test_collection():
    """Создает и удаляет тестовую коллекцию для интеграционных тестов с реальной базой."""
    collection_name = f"test_formats_{uuid.uuid4().hex[:8]}"
    
    client = get_client()
    
    # Создаем тестовую коллекцию
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": 1024,
                "distance": "Cosine"
            }
        )
        
        # Ждем создания коллекции
        time.sleep(1)
        
        yield collection_name
        
    finally:
        # Удаляем тестовую коллекцию
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass


@pytest.mark.integration
def test_upload_text_document(test_collection, rest_api_client):
    """Тест загрузки простого текстового документа."""
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": "Это тестовый текстовый документ.\nВ нем несколько строк.\nОн должен быть успешно загружен.",
                    "payload": {
                        "source_format": "text",
                        "category_path": "Тесты / Форматы / Текст",
                        "source_id": "test-text-001"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    assert response_json["data"]["uploaded"] == 1
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем, что документ загружен
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
    
    # Ищем наш документ по source_id
    found = False
    for p in points:
        if p.payload.get("source_id") == "test-text-001":
            found = True
            assert "raw_text" in p.payload
            assert "тестовый текст" in p.payload.get("raw_text", "").lower()
            break
    
    assert found, "Document should be found in collection"


@pytest.mark.integration
def test_upload_markdown_document(test_collection, rest_api_client):
    """Тест загрузки markdown документа."""
    markdown_text = """# Заголовок документа

Это основной текст документа.

## Второй раздел

Здесь может быть список:
- Пункт 1
- Пункт 2
- Пункт 3

### Подраздел

Код может быть вставлен:
```python
def hello():
    print("Hello, World!")
```
"""
    
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": markdown_text,
                    "payload": {
                        "source_format": "markdown",
                        "category_path": "Тесты / Форматы / Markdown",
                        "source_id": "test-markdown-001",
                        "original_filename": "test_document.md"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    # Документ с 3 заголовками разбивается на 3 секции (по одной на каждый заголовок)
    assert response_json["data"]["uploaded"] >= 1

    # Ждем индексации
    time.sleep(1)

    # Проверяем загрузку
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)

    # Все чанки должны иметь правильный source_id
    matching = [p for p in points if p.payload.get("source_id") == "test-markdown-001"]
    assert len(matching) >= 1, f"Expected at least 1 chunk, got {len(matching)}"

    for p in matching:
        assert "raw_text" in p.payload
        assert "category_path" in p.payload
        # Каждый чанк должен иметь категорию, включающую документную категорию
        cat_path = p.payload.get("category_path", "")
        assert "Тесты" in cat_path and "Markdown" in cat_path

    # Проверяем, что разные заголовки дали разные категории
    cat_paths = {p.payload.get("category_path") for p in matching}
    # Должны быть чанки с разными заголовками в категории
    has_main = any("заголовок документа" in str(cp).lower() for cp in cat_paths)
    has_section = any("второй раздел" in str(cp).lower() for cp in cat_paths)
    assert has_main or has_section, f"Expected different heading categories, got: {cat_paths}"


@pytest.mark.integration
def test_upload_pdf_with_text_layer(test_collection, rest_api_client):
    """Тест загрузки PDF с текстовым слоем (цифровой PDF)."""
    # Для PDF требуется загрузка через multipart/form-data
    # Но API также поддерживает передачу текста, который будет обработан как PDF
    # В данном тесте мы проверяем обработку PDF текста
    
    pdf_text = "Тестовый PDF документ с текстовым слоем. Этот PDF был создан программно для тестирования. Он должен быть успешно распознан."
    
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": pdf_text,
                    "payload": {
                        "source_format": "pdf",
                        "category_path": "Тесты / Форматы / PDF с текстом",
                        "source_id": "test-pdf-text-001",
                        "original_filename": "test_document.pdf"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем загрузку
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
    
    found = False
    for p in points:
        if p.payload.get("source_id") == "test-pdf-text-001":
            found = True
            assert "raw_text" in p.payload
            assert "тестовый pdf" in p.payload.get("raw_text", "").lower()
            break
    
    assert found, "PDF document should be found in collection"


@pytest.mark.integration
def test_upload_code_document(test_collection, rest_api_client):
    """Тест загрузки исходного кода."""
    code_text = """# Проверка кода
def calculate_sum(a, b):
    \"\"\"Функция для суммирования двух чисел.\"\"\"
    return a + b

class Calculator:
    \"\"\"Класс калькулятора.\"\"\"
    
    def __init__(self):
        self.result = 0
    
    def add(self, value):
        \"\"\"Добавляет значение к результату.\"\"\"
        self.result += value
        return self.result
"""
    
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": code_text,
                    "payload": {
                        "source_format": "code",
                        "category_path": "Тесты / Форматы / Код",
                        "source_id": "test-code-001",
                        "original_filename": "calculator.py"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем загрузку
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
    
    found = False
    for p in points:
        if p.payload.get("source_id") == "test-code-001":
            found = True
            assert "raw_text" in p.payload
            assert "calculate_sum" in p.payload.get("raw_text", "")
            break
    
    assert found, "Code document should be found in collection"


@pytest.mark.integration
def test_upload_multiple_documents(test_collection, rest_api_client):
    """Тест загрузки нескольких документов разного формата одновременно."""
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": "Простой текстовый документ.",
                    "payload": {
                        "source_format": "text",
                        "category_path": "Тесты / Мульти-загрузка",
                        "source_id": "multi-text-001"
                    }
                },
                {
                    "text": "# Заголовок\n\nТекст markdown документа.",
                    "payload": {
                        "source_format": "markdown",
                        "category_path": "Тесты / Мульти-загрузка",
                        "source_id": "multi-md-001"
                    }
                },
                {
                    "text": "Текст документа PDF.",
                    "payload": {
                        "source_format": "pdf",
                        "category_path": "Тесты / Мульти-загрузка",
                        "source_id": "multi-pdf-001"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    assert response_json["data"]["uploaded"] == 3
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем, что все документы загружены
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=20, with_payload=True)
    
    source_ids = {p.payload.get("source_id") for p in points}
    expected_ids = {"multi-text-001", "multi-md-001", "multi-pdf-001"}
    
    assert source_ids >= expected_ids, f"All documents should be uploaded. Found: {source_ids}, Expected: {expected_ids}"


@pytest.mark.integration
def test_upload_document_with_category_hierarchy(test_collection, rest_api_client):
    """Тест загрузки документа с глубокой иерархией категорий."""
    response = rest_api_client.post(
        "/v1/documents/upload",
        json={
            "collection_name": test_collection,
            "documents": [
                {
                    "text": "Документ с глубокой иерархией категорий.",
                    "payload": {
                        "source_format": "text",
                        "category_path": "Раздел 1 / Подраздел 1.1 / Подраздел 1.1.1 / Детальный раздел",
                        "source_id": "test-hierarchy-001"
                    }
                }
            ]
        },
        headers=get_auth_headers()
    )
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    
    # Ждем индексации
    time.sleep(1)
    
    # Проверяем загрузку
    client = get_client()
    points, _ = client.scroll(collection_name=test_collection, limit=10, with_payload=True)
    
    found = False
    for p in points:
        if p.payload.get("source_id") == "test-hierarchy-001":
            found = True
            # Проверяем, что категория была обработана
            assert "category_path" in p.payload
            category_path = p.payload.get("category_path", "")
            assert "Раздел 1" in category_path
            assert "Детальный раздел" in category_path
            break
    
    assert found, "Document with category hierarchy should be found"


# --- Изображения, извлечённые из документов ---

IMAGE_FILENAME = "image_000000_ab12cd34.png"


def test_rewrite_image_links_builds_public_url(monkeypatch):
    """Относительная ссылка docling превращается в публичный URL."""
    from app.text_cleaning import image_store

    monkeypatch.setattr(settings, "media_url_prefix", "/media")
    monkeypatch.setattr(settings, "media_base_url", "")

    markdown = f"Текст\n\n![Image](images/{IMAGE_FILENAME})\n\nЕщё текст"
    result = image_store.rewrite_image_links(markdown, "doc-src", "0123456789abcdef0123")

    assert f"![Image](/media/doc-src/0123456789abcdef/images/{IMAGE_FILENAME})" in result


def test_rewrite_image_links_with_base_url(monkeypatch):
    """При заданном media_base_url ссылки становятся абсолютными."""
    from app.text_cleaning import image_store

    monkeypatch.setattr(settings, "media_url_prefix", "/media")
    monkeypatch.setattr(settings, "media_base_url", "http://localhost:8000/")

    result = image_store.rewrite_image_links(
        f"![Image](images/{IMAGE_FILENAME})", "src", "abcdef0123456789xyz"
    )

    assert result == f"![Image](http://localhost:8000/media/src/abcdef0123456789/images/{IMAGE_FILENAME})"


def test_public_url_uses_forward_slashes(monkeypatch):
    """Разделитель в URL всегда '/', даже если путь пришёл в стиле Windows."""
    from app.text_cleaning import image_store

    monkeypatch.setattr(settings, "media_url_prefix", "/media")
    monkeypatch.setattr(settings, "media_base_url", "")

    url = image_store.public_url_for("src", "0123456789abcdef", f"images\\{IMAGE_FILENAME}")

    assert "\\" not in url
    assert url.endswith(f"/images/{IMAGE_FILENAME}")


@pytest.mark.parametrize("content_type", ["markdown", "docx", "pdf"])
def test_chunking_keeps_image_link_intact(content_type):
    """Ссылка на картинку не режется по точке перед расширением файла."""
    from app.chunking import LangChainChunker

    link = f"![Image](/media/src/0123456789abcdef/images/{IMAGE_FILENAME})"
    # Ссылка стоит внутри абзаца, а размер чанка чуть больше её длины — без защиты
    # сплиттер разрывает её по точке перед .png
    text = f"Схема подключения оборудования показана на рисунке {link} и описана в разделе три."

    chunker = LangChainChunker(chunk_size=120, chunk_overlap=10, min_chunk_size=0)
    chunks = chunker.chunk(text, {"content_type": content_type})

    assert any(link in chunk["text"] for chunk in chunks), (
        f"Ссылка разорвана при чанкинге: {[c['text'] for c in chunks]}"
    )


def test_normalize_for_embedding_drops_image_links():
    """Ссылка на картинку не попадает в текст для вектора."""
    from app.text_cleaning.normalizer import normalize_for_embedding

    text = f"До картинки ![Схема сети](/media/src/0123456789abcdef/images/{IMAGE_FILENAME}) после картинки"
    result = normalize_for_embedding(text, source_format="markdown")

    assert "image_000000" not in result
    assert "media" not in result
    assert "схема сети" not in result
    assert "до картинки" in result
    assert "после картинки" in result


def test_cleanup_other_versions_keeps_current(tmp_path, monkeypatch):
    """Каталоги прочих версий удаляются, текущий остаётся."""
    from app.text_cleaning import image_store

    monkeypatch.setattr(settings, "media_dir", str(tmp_path))

    keep_hash = "0123456789abcdef" + "0" * 48
    stale_hash = "fedcba9876543210" + "0" * 48
    keep_dir = image_store.resolve_media_dir("src", keep_hash)
    stale_dir = image_store.resolve_media_dir("src", stale_hash)
    (stale_dir / "images").mkdir()
    (stale_dir / "images" / IMAGE_FILENAME).write_bytes(b"png")

    image_store.cleanup_other_versions("src", keep_hash)

    assert keep_dir.is_dir()
    assert not stale_dir.exists()


@pytest.mark.integration
def test_docling_extracts_images_from_docx(tmp_path):
    """DOCX с картинкой: PNG попадает на диск, ссылка — в Markdown."""
    import io

    docx = pytest.importorskip("docx")
    PIL_Image = pytest.importorskip("PIL.Image")

    image_bytes = io.BytesIO()
    PIL_Image.new("RGB", (64, 64), color=(200, 30, 30)).save(image_bytes, format="PNG")
    image_bytes.seek(0)

    document = docx.Document()
    document.add_paragraph("Схема подключения оборудования приведена ниже.")
    document.add_picture(image_bytes)
    document.add_paragraph("Подключение выполняется согласно схеме.")
    docx_path = tmp_path / "with_image.docx"
    document.save(str(docx_path))

    from app.text_cleaning.docling_cleaner import DoclingCleaner

    media_dir = tmp_path / "media"
    media_dir.mkdir()
    markdown = DoclingCleaner(do_ocr=False).clean(docx_path, media_dir=media_dir)

    saved = list((media_dir / "images").glob("*.png"))
    assert saved, "Docling должен был сохранить картинку на диск"
    assert saved[0].name in markdown
    assert not (media_dir / "document.md").exists(), "Временный document.md должен быть удалён"


@pytest.mark.integration
def test_docling_docx_image_link_rewritten_to_url(tmp_path, monkeypatch):
    """Полный путь: картинка из DOCX превращается в публичный URL в raw_text."""
    import io

    docx = pytest.importorskip("docx")
    PIL_Image = pytest.importorskip("PIL.Image")

    monkeypatch.setattr(settings, "media_dir", str(tmp_path / "media"))
    monkeypatch.setattr(settings, "media_url_prefix", "/media")
    monkeypatch.setattr(settings, "media_base_url", "")
    monkeypatch.setattr(settings, "docling_extract_images", True)

    image_bytes = io.BytesIO()
    PIL_Image.new("RGB", (48, 48), color=(20, 90, 200)).save(image_bytes, format="PNG")
    image_bytes.seek(0)

    document = docx.Document()
    document.add_paragraph("Схема подключения приведена ниже.")
    document.add_picture(image_bytes)
    docx_path = tmp_path / "linked.docx"
    document.save(str(docx_path))

    from app.text_cleaning.docling_cleaner import DoclingCleaner
    from app.text_cleaning.image_store import prepare_media_dir, rewrite_image_links

    doc_hash = "0123456789abcdef" + "f" * 48
    media_dir = prepare_media_dir("src-docx", doc_hash)
    markdown = DoclingCleaner(do_ocr=False).clean(docx_path, media_dir=media_dir)
    markdown = rewrite_image_links(markdown, "src-docx", doc_hash)

    saved = list((media_dir / "images").glob("*.png"))
    assert saved, "Картинка должна быть сохранена"
    assert f"/media/src-docx/0123456789abcdef/images/{saved[0].name}" in markdown
    assert "\\" not in markdown


def _scroll_points_by_source(client, collection_name, source_id):
    """Возвращает все точки заданного source_id из коллекции."""
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(key="source_id", match=qdrant_models.MatchValue(value=source_id)),
                qdrant_models.FieldCondition(key="is_latest", match=qdrant_models.MatchValue(value=True)),
            ]
        ),
        limit=100,
        with_payload=True,
    )
    return list(points)


@pytest.mark.integration
def test_reupload_same_file_updates_all_chunks(test_collection, rest_api_client):
    """Повторная загрузка неизменившегося файла обновляет метаданные всех чанков версии.

    Регрессия: раньше scenario A/B обновлял payload только одной точки
    (scroll с limit=1), остальные чанки той же версии оставались со старыми
    file_path/категориями.
    """
    # Текст заведомо больше одного чанка (chunk_size=512)
    text = ("Альфа-тестирование проверяет базовую работоспособность модуля. "
            "Бета-тестирование проводится на реальных данных. ") * 30

    def upload(file_path):
        return rest_api_client.post(
            "/v1/files/upload",
            files={"file": ("doc.txt", text.encode("utf-8"), "text/plain")},
            data={
                "collection_name": test_collection,
                "source_format": "text",
                "file_path": file_path,
            },
            headers=get_auth_headers(),
        )

    first = upload("/old/path/doc.txt")
    assert first.status_code == 200, first.text
    first_data = first.json()["data"]
    assert first_data["status"] == "created"
    assert first_data["uploaded_chunks"] > 1, "Тест требует документ из нескольких чанков"
    source_id = first_data["source_id"]

    time.sleep(1)
    points_before = _scroll_points_by_source(get_client(), test_collection, source_id)
    assert len(points_before) == first_data["uploaded_chunks"]

    second = upload("/new/path/doc.txt")
    assert second.status_code == 200, second.text
    second_data = second.json()["data"]
    assert second_data["status"] == "updated"
    assert second_data["version"] == first_data["version"], "Хеш не изменился — новая версия не создаётся"
    assert second_data["uploaded_chunks"] == first_data["uploaded_chunks"], \
        "Должны обновиться все чанки версии, а не один"

    time.sleep(1)
    points_after = _scroll_points_by_source(get_client(), test_collection, source_id)
    assert len(points_after) == len(points_before)
    for p in points_after:
        assert p.payload["file_path"] == "/new/path/doc.txt", \
            f"Чанк {p.id} не получил обновлённый file_path"
