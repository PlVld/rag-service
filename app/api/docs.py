import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from starlette.responses import FileResponse
import os

router = APIRouter(tags=["Docs"])
logger = logging.getLogger(__name__)


@router.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
    """Swagger UI с CDN ресурсами."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


@router.get("/redoc", include_in_schema=False)
async def custom_redoc_html() -> HTMLResponse:
    """ReDoc с CDN ресурсами."""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="ReDoc",
    )
