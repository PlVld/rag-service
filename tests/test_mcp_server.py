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

from app.mcp.tools import (
    _format_chunk_meta,
    get_category_hierarchy_tool,
    search_categories_tool,
    search_documents_tool,
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
        mock_client.get_collections = AsyncMock(return_value=type('obj', (object,), {
            'collections': [type('obj', (object,), {'name': 'test'})()]
        })())

        # Добавляем векторы к mock-точкам для корректной работы reranking
        for point in mock_search_points:
            if point.vector is None:
                point.vector = [0.1] * 1024

        with patch("app.utils.basic_normalize", return_value="тест"), \
             patch("app.api.health.get_client", return_value=mock_client), \
             patch("app.mcp_server._perform_search", new_callable=AsyncMock) as mock_search, \
             patch("app.mcp_server.build_qdrant_filter") as mock_filter, \
             patch("app.mcp_server.rerank_by_query_similarity", new_callable=AsyncMock) as mock_rerank, \
             patch("app.mcp_server.search_categories", new_callable=AsyncMock) as mock_sc:

            mock_search.return_value = mock_search_points
            mock_filter.return_value = None
            mock_rerank.return_value = mock_search_points
            mock_sc.return_value = []  # Нет найденных категорий

            result = await _search_documents_internal(
                query_text="тест",
                collection_name="test",
                limit=5,
                group=False
            )

            assert result["success"] is True
            assert len(result["data"]["results"]) == 3
            assert result["data"]["results"][0]["score"] == 0.95
            # Метаданные документа и чанков должны доходить до MCP-слоя
            assert result["data"]["results"][0]["source_id"] == "doc-1"
            assert result["data"]["results"][0]["chunk_index"] == 0
            assert result["data"]["results"][0]["total_chunks"] == 5

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
        mock_client.get_collections = AsyncMock(return_value=type('obj', (object,), {
            'collections': [type('obj', (object,), {'name': 'test'})()]
        })())

        # Добавляем векторы к mock-точкам для корректной работы reranking
        for point in mock_search_points:
            if point.vector is None:
                point.vector = [0.1] * 1024

        with patch("app.utils.basic_normalize", return_value="тест"), \
             patch("app.api.health.get_client", return_value=mock_client), \
             patch("app.mcp_server._perform_search", new_callable=AsyncMock) as mock_search, \
             patch("app.mcp_server.build_qdrant_filter") as mock_filter, \
             patch("app.mcp_server.rerank_by_query_similarity", new_callable=AsyncMock) as mock_rerank:

            mock_search.return_value = mock_search_points
            mock_filter.return_value = MagicMock()
            mock_rerank.return_value = mock_search_points

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
        mock_client.get_collections = AsyncMock(return_value=type('obj', (object,), {
            'collections': [
                type('obj', (object,), {'name': 'documents'})(),
                type('obj', (object,), {'name': 'categories'})(),
            ]
        })())
        mock_client.scroll = AsyncMock(return_value=mock_scroll_result)

        with patch("app.api.categories.search_categories", mock_search_categories), \
             patch("app.utils.basic_normalize", return_value="документация"), \
             patch("app.api.health.get_client", return_value=mock_client), \
             patch("app.api.categories._get_categories_by_facet", new_callable=AsyncMock) as mock_fallback:
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
        assert input_data.limit == 5000

    def test_depth_validation(self):
        """Тест валидации depth."""
        with pytest.raises(Exception):
            GetCategoryHierarchyInput(depth=-1)

        with pytest.raises(Exception):
            GetCategoryHierarchyInput(depth=11)

    def test_limit_validation(self):
        """limit должен быть положительным; null отключает ограничение."""
        assert GetCategoryHierarchyInput(limit=10).limit == 10
        assert GetCategoryHierarchyInput(limit=None).limit is None

        with pytest.raises(Exception):
            GetCategoryHierarchyInput(limit=0)

        with pytest.raises(Exception):
            GetCategoryHierarchyInput(limit=-1)


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


class TestGetCategoryHierarchyTool:
    """Тесты MCP-инструмента иерархии категорий."""

    @pytest.mark.asyncio
    async def test_passes_limit_to_internal(self):
        with patch(
            "app.mcp.tools._get_category_hierarchy_internal",
            new_callable=AsyncMock,
        ) as mock_internal:
            mock_internal.return_value = {"success": True, "data": {"results": []}}

            await get_category_hierarchy_tool(
                GetCategoryHierarchyInput(collection_name="documents", limit=42)
            )

        assert mock_internal.await_args.kwargs["limit"] == 42

    @pytest.mark.asyncio
    async def test_reports_truncated_output(self):
        data = {
            "results": [
                {
                    "name": "coll1",
                    "categories": [{"category_path": "A", "chunk_count": 1}],
                    "total_categories": 3,
                    "truncated": True,
                },
                {
                    "name": "coll2",
                    "categories": [],
                    "total_categories": 2,
                    "truncated": True,
                },
            ],
            "limit": 1,
            "truncated": True,
            "total_categories": 5,
            "shown_categories": 1,
        }
        with patch(
            "app.mcp.tools._get_category_hierarchy_internal",
            new_callable=AsyncMock,
            return_value={"success": True, "data": data},
        ):
            output = await get_category_hierarchy_tool(GetCategoryHierarchyInput())

        assert "показано 1 из 3 категорий" in output
        assert "категории не показаны из-за лимита: всего 2" in output
        assert "Показано 1 из 5 категорий (limit=1)" in output

    @pytest.mark.asyncio
    async def test_does_not_report_limit_without_truncation(self):
        data = {
            "results": [
                {
                    "name": "coll1",
                    "categories": [{"category_path": "A", "chunk_count": 1}],
                    "total_categories": 1,
                    "truncated": False,
                }
            ],
            "limit": 5000,
            "truncated": False,
            "total_categories": 1,
            "shown_categories": 1,
        }
        with patch(
            "app.mcp.tools._get_category_hierarchy_internal",
            new_callable=AsyncMock,
            return_value={"success": True, "data": data},
        ):
            output = await get_category_hierarchy_tool(GetCategoryHierarchyInput())

        assert "- A (1 чанков)" in output
        assert "limit=" not in output
        assert "показано" not in output


class TestSearchDocumentsToolOutput:
    """Тесты формата ответа MCP-инструмента поиска документов."""

    @pytest.mark.asyncio
    async def test_grouped_result_shows_source_id_and_chunk_range(self):
        """Для group=true выводятся source_id и диапазон чанков."""
        grouped_result = {
            "success": True,
            "data": {
                "results": [{
                    "id": None,
                    "score": 0.85,
                    "document": "Текст фрагмента",
                    "payload": None,
                    "collection_name": "documents",
                    "category_path": "Документация / API",
                    "source_id": "doc-1",
                    "chunk_index": None,
                    "first_chunk_index": 2,
                    "last_chunk_index": 4,
                    "total_chunks": 10,
                }]
            },
        }

        with patch(
            "app.mcp.tools._search_documents_internal",
            new_callable=AsyncMock,
            return_value=grouped_result,
        ):
            output = await search_documents_tool(SearchDocumentsInput(query_text="тест"))

        assert "source_id=doc-1" in output
        assert "collection_name=documents" in output
        assert "category_path=Документация / API" in output
        assert "чанки 2–4 из 10" in output
        assert "Текст фрагмента" in output

    @pytest.mark.asyncio
    async def test_individual_result_shows_chunk_index(self):
        """Для group=false выводятся source_id и номер чанка."""
        individual_result = {
            "success": True,
            "data": {
                "results": [{
                    "id": None,
                    "score": 0.75,
                    "document": "Фрагмент чанка",
                    "payload": None,
                    "collection_name": "documents",
                    "category_path": "Документация / API",
                    "source_id": "doc-2",
                    "chunk_index": 1,
                    "first_chunk_index": None,
                    "last_chunk_index": None,
                    "total_chunks": 8,
                }]
            },
        }

        with patch(
            "app.mcp.tools._search_documents_internal",
            new_callable=AsyncMock,
            return_value=individual_result,
        ):
            output = await search_documents_tool(
                SearchDocumentsInput(query_text="тест", group=False)
            )

        assert "source_id=doc-2" in output
        assert "collection_name=documents" in output
        assert "category_path=Документация / API" in output
        assert "чанк 1 из 8" in output

    @pytest.mark.asyncio
    async def test_missing_metadata_does_not_break_output(self):
        """Если метаданных нет, вывод остаётся прежним."""
        result_without_meta = {
            "success": True,
            "data": {
                "results": [{
                    "id": None,
                    "score": 0.5,
                    "document": "Текст",
                    "payload": None,
                    "collection_name": "documents",
                    "category_path": "Документация / API",
                    "source_id": None,
                    "chunk_index": None,
                    "first_chunk_index": None,
                    "last_chunk_index": None,
                    "total_chunks": None,
                }]
            },
        }

        with patch(
            "app.mcp.tools._search_documents_internal",
            new_callable=AsyncMock,
            return_value=result_without_meta,
        ):
            output = await search_documents_tool(SearchDocumentsInput(query_text="тест"))

        assert (
            "1. [score=0.500] (collection_name=documents, "
            "category_path=Документация / API) Текст"
        ) in output

    def test_category_path_list_is_joined(self):
        """category_path-список выводится через ' / '."""
        meta = _format_chunk_meta({
            "collection_name": "documents",
            "category_path": ["Документация", "API"],
            "source_id": "doc-1",
        })

        assert meta == (
            " (collection_name=documents, "
            "category_path=Документация / API, source_id=doc-1)"
        )

    def test_exact_match_and_phrases_are_rendered(self):
        """Exact-совпадение и найденные фразы видны в метаданных."""
        meta = _format_chunk_meta({
            "collection_name": "1c_helpdesk",
            "category_path": "1С / Синтаксис",
            "exact_match": True,
            "matched_phrases": ["ЕСТЬNULL", "ISNULL"],
        })

        assert "exact_match=true" in meta
        assert "matched_phrases=ЕСТЬNULL, ISNULL" in meta