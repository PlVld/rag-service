"""
Объединяющий модуль для эндпоинтов документов.
Импортирует эндпоинты из подмодулей для backward compatibility.
"""
from app.api.documents_upload import router as upload_router, upload_documents, process_documents
from app.api.documents_search import router as search_router, search_documents, rerank_by_query_similarity, _perform_search, _group_search_results, _return_individual_chunks
from app.api.documents_utils import (
    build_qdrant_filter,
    get_category_path_from_payload,
    get_all_chunks,
    _matches_category_path,
    _normalize_category_path,
    _get_documents_upload_request_from_form_or_json,
    _prepare_search_context,
)

# Для backward compatibility используем search_router как основной роутер
# Это позволяет использовать documents.router в main.py без изменений
router = search_router

__all__ = [
    # Routers
    'upload_router',
    'search_router',
    'router',
    # Endpoints
    'upload_documents',
    'search_documents',
    # Functions
    'process_documents',
    'rerank_by_query_similarity',
    '_perform_search',  # добавлен для mcp_server.py
    '_group_search_results',
    '_return_individual_chunks',
    # Utils
    'build_qdrant_filter',
    'get_category_path_from_payload',
    'get_all_chunks',
    '_matches_category_path',
    '_normalize_category_path',
    '_get_documents_upload_request_from_form_or_json',
    '_prepare_search_context',
]
