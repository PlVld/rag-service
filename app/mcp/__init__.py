# app/mcp — MCP-прокси, инструменты и middlewares.
#
# Структура:
#   tools.py       — MCP-инструменты и явный реестр TOOLS
#   middlewares.py — middlewares для /mcp-эндпоинтов
#   proxy.py       — JSON-RPC прокси и регистрация маршрутов

from app.mcp.proxy import register_mcp_routes, mcp_app
from app.mcp.middlewares import register_mcp_middlewares
from app.mcp.tools import TOOLS

__all__ = ["register_mcp_routes", "register_mcp_middlewares", "mcp_app", "TOOLS"]
