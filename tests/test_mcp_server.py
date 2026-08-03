"""Тесты для MCP сервера и его инструментов."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.mcp_server import (
    SearchDocumentsInput,
    SearchCategoriesInput,
    GetCategoryHierarchyInput,
    _search_documents_internal,
    _get_category_hierarchy_internal,
)

from app.main import (
    search_categories_tool,
)


class TestSearchDocumentsInput:
    """Тесты валидации входных данных для поиска документов."""

    def test_valid_input(self):
        """Тест корректных входных данных."""
        input_data = SearchDocumentsInput(
            query_text="Как настроить API?",
            collection_name="documents",
            limit=5,
        )
        assert input_data.query_text == "Как настроить API?"
        assert input_data.collection_name == "documents"
        assert input_data.limit == 5
        assert input_data.filter_criteria is None
        assert input_data.include_old_versions is False

    def test_default_values(self):
        """Тест значений по умолчанию."""
        input_data = SearchDocumentsInput(
            query_text="тест",
            collection_name="documents"
        )
        assert input_data.limit == 10
        assert input_data.filter_criteria is None
        assert input_data.include_old_versions is False

    def test_limit_validation(self):
        """Тест валидации limit."""
        with pytest.raises(Exception):
            SearchDocumentsInput(
                query_text="тест",
                collection_name="documents",
                limit=0
            )

        with pytest.raises(Exception):
            SearchDocumentsInput(
                query_text="тест",
                collection_name="documents",
                limit=101
            )

    def test_filter_validation(self):
        """Тест валидации фильтра."""
        input_data = SearchDocumentsInput(
            query_text="тест",
            collection_name="documents",
            filter={"source_format": "text", "is_latest": True}
        )
        # Доступ через alias "filter" сохраняется, но поле называется filter_criteria
        assert input_data.filter_criteria["source_format"] == "text"


class TestSearchCategoriesInput:
    """Тесты валидации входных данных для поиска категорий."""

    def test_valid_input(self):
        """Тест корректных входных данных."""
        input_data = SearchCategoriesInput(
            query_text="Документация по API",
            limit=5
        )
        assert input_data.query_text == "Документация по API"
        assert input_data.limit == 5

    def test_default_limit(self):
        """Тест значения по умолчанию для limit."""
        input_data = SearchCategoriesInput(query_text="тест")
        assert input_data.limit == 10

    def test_limit_validation(self):
        """Тест валидации limit."""
        with pytest.raises(Exception):
            SearchCategoriesInput(query_text="тест", limit=0)

        with pytest.raises(Exception):
            SearchCategoriesInput(query_text="тест", limit=51)


class TestSearchDocumentsInternal:
    """Тесты внутренней функции поиска документов."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_results(self, mock_embeddings, mock_search_categories):
        """Тест пустого запроса."""
        with patch("app.utils.basic_normalize", return_value=""):
            result = await _search_documents_internal(
                query_text="",
                collection_name="documents"
            )
        assert result["success"] is True
        assert result["data"]["results"] == []

    @pytest.mark.asyncio
    async def test_successful_search(
        self,
        mock_embeddings,
        mock_search_categories,
        mock_search_points
    ):
        """Тест успешного поиска."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = type('obj', (object,), {
            'collections': [type('obj', (object,), {'name': 'test'})()]
        })()
        
        with patch("app.utils.basic_normalize", return_value="тест"), \
             patch("app.api.health.get_client", return_value=mock_client), \
             patch("app.mcp_server._perform_search", new_callable=AsyncMock) as mock_search, \
             patch("app.mcp_server.build_qdrant_filter") as mock_filter:

            mock_search.return_value = mock_search_points
            mock_filter.return_value = None

            result = await _search_documents_internal(
                query_text="тест",
                collection_name="test",
                limit=5
            )

            assert result["success"] is True
            assert len(result["data"]["results"]) == 3
            assert result["data"]["results"][0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_search_collection_not_found(self, mock_embeddings, mock_search_categories):
        """Тест ошибки - коллекция не найдена."""
        with patch("app.utils.basic_normalize", return_value="тест"), \
             patch("app.api.health.get_client") as mock_client, \
             patch("app.api.documents._perform_search", new_callable=AsyncMock) as mock_search, \
             patch("app.api.documents.build_qdrant_filter") as mock_filter:

            mock_search.side_effect = ValueError("Collection 'test' not found")
            mock_filter.return_value = None

            result = await _search_documents_internal(
                query_text="тест",
                collection_name="test"
            )

            assert result["success"] is False
            assert result["error"]["code"] == "collection_not_found"

    @pytest.mark.asyncio
    async def test_search_with_filter(
        self,
        mock_embeddings,
        mock_search_categories,
        mock_search_points
    ):
        """Тест поиска с фильтром."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = type('obj', (object,), {
            'collections': [type('obj', (object,), {'name': 'test'})()]
        })()
        
        with patch("app.utils.basic_normalize", return_value="тест"), \
             patch("app.api.health.get_client", return_value=mock_client), \
             patch("app.mcp_server._perform_search", new_callable=AsyncMock) as mock_search, \
             patch("app.mcp_server.build_qdrant_filter") as mock_filter:

            mock_search.return_value = mock_search_points
            mock_filter.return_value = MagicMock()

            # Используем filter_criteria вместо filter
            result = await _search_documents_internal(
                query_text="тест",
                collection_name="test",
                filter_criteria={"source_format": "markdown"}
            )

            assert result["success"] is True
            mock_filter.assert_called_once()


class TestSearchCategoriesTool:
    """Тесты инструмента поиска категорий."""

    @pytest.mark.asyncio
    async def test_categories_found(self, mock_embeddings, mock_collections, mock_facet_result):
        """Тест успешного поиска категорий."""
        from unittest.mock import AsyncMock, MagicMock
        
        # Подготовим асинхронный мок для search_categories_by_collections
        mock_search_categories_by_collections = AsyncMock()
        mock_search_categories_by_collections.return_value = {
            "documents": [
                MagicMock(
                    id="cat-1",
                    score=0.95,
                    category_name="API",
                    category_path="Документация / API",
                    categories=["Документация", "API"],
                    category_level=1,
                    category_id="cat-id-1",
                    levels={"category_level0": "Документация", "category_level1": "API"},
                    id_levels={"category_id_level0": "id-0", "category_id_level1": "id-1"},
                ),
            ]
        }
        
        with patch("app.api.categories.search_categories_by_collections", mock_search_categories_by_collections):
            input_data = SearchCategoriesInput(
                query_text="Документация",
                limit=5
            )

            result = await search_categories_tool(input_data=input_data)

            assert "Релевантные категории по коллекциям:" in result
            assert "Документация / API" in result
            assert "Коллекция: documents" in result
            assert "score:" in result

    @pytest.mark.asyncio
    async def test_no_categories_found(self, mock_embeddings):
        """Тест когда категории не найдены."""
        with patch("app.api.categories.search_categories_by_collections", new_callable=AsyncMock) as mock:
            mock.return_value = {}

            input_data = SearchCategoriesInput(
                query_text="несуществующая категория",
                limit=5
            )

            result = await search_categories_tool(input_data=input_data)
            assert result == "Категории не найдены."


class TestSearchCategoriesByCollections:
    """Тесты функции группировки категорий по коллекциям."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, mock_embeddings):
        """Тест пустого запроса."""
        from app.api.categories import search_categories_by_collections
        
        result = await search_categories_by_collections(query_text="", limit=10)
        
        assert result == {}

    @pytest.mark.asyncio
    async def test_successful_grouped_search(
        self,
        mock_embeddings
    ):
        """Тест успешного группированного поиска."""
        from app.api.categories import search_categories_by_collections
        from unittest.mock import AsyncMock, MagicMock
        
        # Подготовим асинхронный мок для search_categories
        mock_search_categories = AsyncMock()
        mock_search_categories.return_value = [
            MagicMock(
                id="cat-1",
                score=0.95,
                category_name="API",
                category_path="Документация / API",
                categories=["Документация", "API"],
                category_level=1,
                category_id="cat-id-1",
                levels={"category_level0": "Документация", "category_level1": "API"},
                id_levels={"category_id_level0": "id-0", "category_id_level1": "id-1"},
            ),
            MagicMock(
                id="cat-2",
                score=0.85,
                category_name="Настройка",
                category_path="Документация / Настройка",
                categories=["Документация", "Настройка"],
                category_level=1,
                category_id="cat-id-2",
                levels={"category_level0": "Документация", "category_level1": "Настройка"},
                id_levels={"category_id_level0": "id-0", "category_id_level1": "id-2"},
            ),
        ]
        
        # Подготовим мок для fallback scroll
        mock_scroll_result = (
            [
                MagicMock(
                    payload={
                        "category_path": ["Документация", "API"],
                        "source_id": "doc-1"
                    }
                ),
                MagicMock(
                    payload={
                        "category_path": ["Документация", "Настройка"],
                        "source_id": "doc-2"
                    }
                ),
                MagicMock(
                    payload={
                        "category_path": ["Документация", "API"],
                        "source_id": "doc-3"
                    }
                ),
            ],
            None  # offset
        )
        
        # Подготовим мокированный клиент
        mock_client = MagicMock()
        mock_client.get_collections.return_value = type('obj', (object,), {
            'collections': [
                type('obj', (object,), {'name': 'documents'})(),
                type('obj', (object,), {'name': 'categories'})(),
            ]
        })()
        mock_client.scroll.return_value = mock_scroll_result
        
        with patch("app.api.categories.search_categories", mock_search_categories), \
             patch("app.utils.basic_normalize", return_value="документация"), \
             patch("app.api.health.get_client", return_value=mock_client) as mock_get_client, \
             patch("app.api.categories._get_categories_by_facet", new_callable=AsyncMock) as mock_fallback:
            print(f"DEBUG: mock_get_client.return_value = {mock_get_client.return_value}")
            print(f"DEBUG: mock_get_client.return_value.get_collections = {mock_get_client.return_value.get_collections}")
            print(f"DEBUG: mock_get_client.return_value.get_collections() = {mock_get_client.return_value.get_collections()}")
            print(f"DEBUG: mock_get_client.return_value.get_collections().collections = {mock_get_client.return_value.get_collections().collections}")
            mock_fallback.return_value = {
                "Документация / API": 2,
                "Документация / Настройка": 1
            }
            result = await search_categories_by_collections(query_text="документация", limit=10)
        
        print(f"DEBUG: result = {result}")
        print(f"DEBUG: result.keys() = {result.keys()}")
        
        assert isinstance(result, dict)
        # Проверяем, что результат содержит хотя бы одну коллекцию с категориями
        assert len(result) > 0
        # Проверяем, что хотя бы в одной коллекции есть категории
        has_categories = False
        for collection_name, categories in result.items():
            if len(categories) > 0:
                has_categories = True
                assert categories[0].category_path == "Документация / API"
                break
        assert has_categories, "Expected at least one collection with categories"


class TestGetCategoryHierarchyInput:
    """Тесты валидации входных данных для получения иерархии категорий."""

    def test_valid_input_with_collection(self):
        """Тест корректных входных данных с указанием коллекции."""
        input_data = GetCategoryHierarchyInput(
            collection_name="documents",
            depth=2,
            categories=["Документация / API", "Документация / Настройка"]
        )
        assert input_data.collection_name == "documents"
        assert input_data.depth == 2
        assert input_data.categories == ["Документация / API", "Документация / Настройка"]

    def test_valid_input_without_collection(self):
        """Тест корректных входных данных без указания коллекции."""
        input_data = GetCategoryHierarchyInput()
        assert input_data.collection_name is None
        assert input_data.depth == 1
        assert input_data.categories is None

    def test_default_values(self):
        """Тест значений по умолчанию."""
        input_data = GetCategoryHierarchyInput()
        assert input_data.collection_name is None
        assert input_data.depth == 1
        assert input_data.categories is None

    def test_depth_validation(self):
        """Тест валидации depth."""
        with pytest.raises(Exception):
            GetCategoryHierarchyInput(depth=-1)

        with pytest.raises(Exception):
            GetCategoryHierarchyInput(depth=11)


class TestGetCategoryHierarchyInternal:
    """Тесты внутренней функции получения иерархии категорий."""

    @pytest.mark.asyncio
    async def test_empty_categories_list(self, mock_embeddings):
        """Тест пустого списка категорий."""
        with patch("app.api.admin._get_category_hierarchy_data", new_callable=AsyncMock) as mock_data:
            mock_data.return_value = {"results": []}
            
            result = await _get_category_hierarchy_internal()
            
            assert result["success"] is True
            assert result["data"]["results"] == []

    @pytest.mark.asyncio
    async def test_successful_hierarchy(self, mock_embeddings):
        """Тест успешного получения иерархии."""
        with patch("app.api.admin._get_category_hierarchy_data", new_callable=AsyncMock) as mock_data:
            mock_data.return_value = {
                "results": [
                    {
                        "name": "documents",
                        "categories": [
                            {
                                "category_path": "Документация / API",
                                "chunk_count": 10
                            },
                            {
                                "category_path": "Документация / Настройка",
                                "chunk_count": 5
                            }
                        ]
                    }
                ]
            }
            
            result = await _get_category_hierarchy_internal(
                collection_name="documents",
                depth=2
            )
            
            assert result["success"] is True
            assert len(result["data"]["results"]) == 1
            assert result["data"]["results"][0]["name"] == "documents"
            assert len(result["data"]["results"][0]["categories"]) == 2

    @pytest.mark.asyncio
    async def test_collection_not_found(self, mock_embeddings):
        """Тест ошибки - коллекция не найдена."""
        with patch("app.api.admin._get_category_hierarchy_data", new_callable=AsyncMock) as mock_data:
            mock_data.side_effect = ValueError("Collection 'test' not found")
            
            result = await _get_category_hierarchy_internal(collection_name="test")
            
            assert result["success"] is False
            assert result["error"]["code"] == "collection_not_found"

    @pytest.mark.asyncio
    async def test_hierarchy_failed(self, mock_embeddings):
        """Тест ошибки - неожиданная ошибка."""
        with patch("app.api.admin._get_category_hierarchy_data", new_callable=AsyncMock) as mock_data:
            mock_data.side_effect = Exception("Unknown error")
            
            result = await _get_category_hierarchy_internal()
            
            assert result["success"] is False
            assert result["error"]["code"] == "hierarchy_failed"
