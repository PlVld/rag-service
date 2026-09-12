# app/mcp/middlewares.py
# Middlewares для MCP-эндпоинтов: нормализация пути/заголовков, авторизация,
# ограничение инструментов и отладочное логирование.

import json
import logging
import secrets as _secrets

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from app.core.config import settings
from app.mcp.tools import TOOLS

Response = StarletteResponse

diag_logger = logging.getLogger("rag_service.diagnostics")


class MCPPathNormalizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.scope.get("path", "")
        if path in ("/mcp", "/mcp/rpc", "/mcp/rpc/"):
            request.scope["path"] = "/mcp/"
        return await call_next(request)


class MCPAllowedToolsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/mcp") and request.method.upper() == "POST":
            allowed = settings.allowed_mcp_tools_set
            if allowed:
                body = await request.body()
                try:
                    j = json.loads(body.decode('utf-8') if isinstance(body, (bytes, bytearray)) else body)
                    method = j.get("method")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    method = None

                if method and method in TOOLS and method not in allowed:
                    diag_logger.warning("MCP call to forbidden tool: %s allowed=%s registered=%s",
                                        method, allowed, set(TOOLS))
                    return Response(status_code=403, content=f"MCP tool '{method}' is not allowed")

                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive
        return await call_next(request)


class MCPNormalizeHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/mcp") and request.method.upper() == "POST":
            scope_headers = list(request.scope.get("headers", []))
            header_names = {h[0].decode('latin1').lower(): i for i, h in enumerate(scope_headers)}

            def set_header(name: str, value: str):
                key = name.lower().encode('latin1')
                val = value.encode('latin1')
                if name.lower() in header_names:
                    scope_headers[header_names[name.lower()]] = (key, val)
                else:
                    scope_headers.append((key, val))

            set_header('accept', 'application/json')
            if 'content-type' not in header_names:
                set_header('content-type', 'application/json')
            request.scope['headers'] = scope_headers
        return await call_next(request)


class MCPAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/mcp"):
            auth = request.headers.get("authorization")
            if not auth or not auth.lower().startswith("bearer "):
                return Response(status_code=401, content="Authentication required",
                                headers={"WWW-Authenticate": "Bearer"})
            try:
                bearer_token = auth.split(" ", 1)[1]
                diag_logger.info("MCP Auth attempt (token не логируется)")
            except (ValueError, TypeError, IndexError):
                return Response(status_code=401, content="Invalid authentication",
                                headers={"WWW-Authenticate": "Bearer"})

            if not (_secrets.compare_digest(bearer_token, settings.rag_service_api_key)):
                return Response(status_code=401, content="Invalid credentials",
                                headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)


class MCPLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not request.scope.get("path", "").startswith("/mcp"):
            return await call_next(request)
        try:
            body = await request.body()

            async def receive():
                return {"type": "http.request", "body": body}

            request._receive = receive

            headers = {k: ("<REDACTED>" if k.lower() == "authorization" else v) for k, v in request.headers.items()}
            diag_logger.info(
                "[MCP-DEBUG] Request incoming: method=%s original_path=%s normalized_path=%s headers=%s body_preview=%s",
                request.method, request.scope.get("raw_path", request.scope.get("path")), request.scope.get("path"),
                headers, (body[:500] if body else b""))

            response = await call_next(request)

            resp_body = b""
            if hasattr(response, "body"):
                body_bytes = getattr(response, "body", None)
                if body_bytes is not None:
                    resp_body = body_bytes if isinstance(body_bytes, bytes) else (body_bytes.encode('utf-8') if isinstance(body_bytes, str) else b"")
            if not resp_body and hasattr(response, "body_iterator"):
                try:
                    async for chunk in response.body_iterator:
                        resp_body += chunk if isinstance(chunk, bytes) else chunk.encode('utf-8')
                except (TypeError, AttributeError):
                    pass

            diag_logger.info("[MCP-DEBUG] Response outgoing: status=%s path=%s resp_preview=%s headers=%s",
                             response.status_code, request.scope.get("path"), (resp_body[:500] if resp_body else b""),
                             dict(response.headers))

            if resp_body:
                new_resp = StarletteResponse(content=resp_body, status_code=response.status_code,
                                             headers=dict(response.headers), media_type=response.media_type)
                return new_resp
            return response
        except (AttributeError, TypeError, KeyError) as e:
            diag_logger.exception("[MCP-DEBUG] Logging middleware error: %s", e)
            return await call_next(request)


def register_mcp_middlewares(app) -> None:
    """Подключает MCP-middlewares к приложению (в порядке, обратном порядку обработки)."""
    app.add_middleware(MCPLoggingMiddleware)
    app.add_middleware(MCPAllowedToolsMiddleware)
    app.add_middleware(MCPPathNormalizeMiddleware)
    app.add_middleware(MCPNormalizeHeadersMiddleware)
    if settings.mcp_auth_enabled:
        app.add_middleware(MCPAuthMiddleware)
