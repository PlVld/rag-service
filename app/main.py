import logging
import time
import uvicorn
import json
from fastapi import FastAPI, Request
from pydantic import ValidationError
from app.api.health import get_client
from fastapi_mcp import FastApiMCP
from starlette.middleware.base import BaseHTTPMiddleware
from app.api import documents, health, categories, files, admin
from app.core.config import settings

# Настройка логирования из конфигурации
settings.configure_logging()
# Уменьшаем уровень для шумных библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

diag_logger = logging.getLogger("rag_service.diagnostics")

# --- Middleware для обработки "Expect: 100-continue" + "Transfer-Encoding: chunked" ---
_BODYLESS_METHODS = {"GET", "HEAD", "OPTIONS", "DELETE"}


class ConsumeRequestBodyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if (request.method in _BODYLESS_METHODS
                and "chunked" in request.headers.get("transfer-encoding", "")):
            await request.body()
        return await call_next(request)


class RequestDiagnosticsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client = request.client
        client_info = f"{client.host}:{client.port}" if client else "unknown"
        method = request.method
        url = str(request.url)
        headers = dict(request.headers)

        diag_logger.info(
            "Incoming request: client=%s method=%s url=%s headers=%s",
            client_info, method, url, headers,
        )

        start = time.time()
        try:
            response = await call_next(request)
            elapsed = time.time() - start
            diag_logger.info(
                "Response: client=%s method=%s url=%s status=%s time=%.3fs",
                client_info, method, url, response.status_code, elapsed,
            )
            return response
        except Exception as exc:
            elapsed = time.time() - start
            diag_logger.error(
                "Request failed: client=%s method=%s url=%s error=%s time=%.3fs",
                client_info, method, url, exc, elapsed,
            )
            raise


app = FastAPI(
    title="Universal Document Vector Search Service",
    description="Микросервис для векторного поиска по документам с поддержкой метаданных, категорий и версионирования",
    version="1.2.0"
)

app.add_middleware(RequestDiagnosticsMiddleware)
app.add_middleware(ConsumeRequestBodyMiddleware)

# Подключаем роутеры
app.include_router(documents.upload_router)
app.include_router(documents.search_router)
app.include_router(health.router)
app.include_router(categories.router)
app.include_router(files.router)
app.include_router(admin.router)

# --- Инициализация MCP ---
import base64
import secrets as _secrets
from starlette.responses import Response


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

                registered_tools = set()
                try:
                    import app.mcp_server as mcpmodule
                    for name, obj in vars(mcpmodule).items():
                        if callable(obj) and name.endswith("_tool"):
                            registered_tools.add(name)
                except (ImportError, AttributeError):
                    registered_tools = set()

                if method and registered_tools and method in registered_tools and method not in allowed:
                    diag_logger.warning("MCP call to forbidden tool: %s allowed=%s registered=%s", method, allowed,
                                        registered_tools)
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
                diag_logger.info("MCP Auth attempt: token='%s...'", bearer_token[:8])
            except (ValueError, TypeError, IndexError):
                return Response(status_code=401, content="Invalid authentication",
                                headers={"WWW-Authenticate": "Bearer"})

            if not (_secrets.compare_digest(bearer_token, settings.rag_service_api_key)):
                return Response(status_code=401, content="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
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


app.add_middleware(MCPLoggingMiddleware)


class MCPPathNormalizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.scope.get("path", "")
        if path in ("/mcp", "/mcp/rpc", "/mcp/rpc/"):
            request.scope["path"] = "/mcp/"
        return await call_next(request)


app.add_middleware(MCPAllowedToolsMiddleware)
app.add_middleware(MCPPathNormalizeMiddleware)
app.add_middleware(MCPNormalizeHeadersMiddleware)

if settings.mcp_auth_enabled:
    app.add_middleware(MCPAuthMiddleware)

# --- FastApiMCP (используется только для fallback методов initialize/ping) ---
from fastapi import FastAPI as _FastAPI
from app.mcp_server import SearchDocumentsInput, SearchCategoriesInput, GetCategoryHierarchyInput
from app.mcp_server import _search_documents_internal, _search_categories_internal, _get_category_hierarchy_internal
mcp_app = _FastAPI(title="MCP subapp")
mcp = FastApiMCP(
    mcp_app,
    name="Document Search MCP",
    description="Document semantic search service.",
    auth_config=None,
)


# --- Определение функций MCP-инструментов (используются в кастомном прокси) ---
async def search_documents_tool(input_data: SearchDocumentsInput) -> str:
    """Поиск соответствующих тексту запроса чанков.
    
    По умолчанию результаты группируются по category_path (group=true).
    Чтобы получить отдельные чанки, установите group=false.
    """
    result = await _search_documents_internal(
        query_text=input_data.query_text,
        collection_name=input_data.collection_name,
        limit=input_data.limit,
        filter_criteria=input_data.filter_criteria,
        include_old_versions=input_data.include_old_versions,
        max_text_length=input_data.max_text_length,
        group=input_data.group,
    )

    if result["success"]:
        results = result["data"]["results"]
        if not results:
            return "По вашему запросу ничего не найдено."

        output = f"Найдено {len(results)} результатов:\n\n"
        for i, r in enumerate(results, 1):
            # Используем подготовленный snippet из mcp_server (уже с учетом max_text_length)
            snippet = r["document"]
            output += f"{i}. [score={r['score']:.3f}] {snippet}\n\n"
        return output
    else:
        return f"Ошибка поиска: {result['error']['message']}"


async def search_categories_tool(input_data: SearchCategoriesInput) -> str:
    """Поиск категорий документов, соответствующих текстовому запросу.
    Категории возвращаются сгруппированными по коллекциям.
    """
    # Импорт search_categories_by_collections в начале функции
    from app.api.categories import search_categories_by_collections
    
    result = await _search_categories_internal(
        query_text=input_data.query_text,
        limit=input_data.limit,
        fields=input_data.fields,
    )
    
    if not result["success"]:
        return f"Ошибка поиска категорий: {result.get('error', 'Неизвестная ошибка')}"
    
    categories_res = await search_categories_by_collections(input_data.query_text, input_data.limit)

    if not categories_res:
        return "Категории не найдены."

    output = "Релевантные категории по коллекциям:\n\n"
    for collection_name, cat_list in categories_res.items():
        output += f"### Коллекция: {collection_name}\n"
        for cat in cat_list:
            output += f"- {cat.category_path} (score: {cat.score:.3f})\n"
        output += "\n"
    return output


async def get_category_hierarchy_tool(input_data: GetCategoryHierarchyInput) -> str:
    """Получение иерархии категорий с количеством чанков для указанных коллекций.
    Возвращает категории с количеством чанков, организованные по коллекциям.
    
    Полезно для:
    - Просмотра доступных категорий в коллекциях
    - Проверки структуры категорий
    - Получения количества документов в каждой категории
    """
    result = await _get_category_hierarchy_internal(
        collection_name=input_data.collection_name,
        depth=input_data.depth,
        categories=input_data.categories,
    )

    if result["success"]:
        results = result["data"].get("results", [])
        if not results:
            return "Иерархия категорий не найдена."

        output = "Иерархия категорий:\n\n"
        for coll in results:
            coll_name = coll.get("name", "unknown")
            categories_list = coll.get("categories", [])
            
            if not categories_list:
                output += f"### Коллекция: {coll_name}\n   (категории не найдены)\n\n"
            else:
                output += f"### Коллекция: {coll_name}\n"
                for cat in categories_list:
                    cat_path = cat.get("category_path", "unknown")
                    chunk_count = cat.get("chunk_count", 0)
                    output += f"- {cat_path} ({chunk_count} чанков)\n"
                output += "\n"
        return output
    else:
        return f"Ошибка получения иерархии категорий: {result['error']['message']}"


# --- Простое кэширование информации о коллекциях (TTL 60 секунд) ---
_collections_cache: dict = {"data": None, "timestamp": 0}
_CACHE_TTL: int = 60


async def get_collections_info() -> str:
    """Возвращает строку с информацией о доступных коллекциях и их корневых категориях (с кэшированием через facet)."""
    import time
    now = time.time()
    if _collections_cache["data"] is not None and (now - _collections_cache["timestamp"]) < _CACHE_TTL:
        return _collections_cache["data"]

    try:
        client = get_client()
        collections_result = client.get_collections()
        collections = collections_result.collections if hasattr(collections_result, 'collections') else []

        collection_info = []
        for collection in collections:
            if collection.name in ["categories"]:
                continue
            try:
                # Используем facet для получения уникальных category_level0
                # Поле должно быть ключевым (keyword) в Qdrant
                facet_result = client.facet(
                    collection_name=collection.name,
                    key="category_level0",      # поле, где хранится корневая категория
                    limit=5                    # максимум уникальных значений
                )
                # facet_result - это объект с полями: hits (список значений и count)
                if hasattr(facet_result, 'hits'):
                    root_categories = [hit.value for hit in facet_result.hits if hit.value]
                else:
                    root_categories = []

                # Отображаем не более 5 категорий для краткости
                displayed = root_categories[:5]
                summary = ", ".join(displayed)
                if len(root_categories) > 5:
                    summary += f" и ещё {len(root_categories)-5}"
                collection_info.append(f"{collection.name} [{summary}]")
            except Exception as e:
                diag_logger.error(f"Ошибка получения категорий для {collection.name}: {e}")

        if collection_info:
            result = "\nКоллекции [категории]:\n" + "\n".join(collection_info)
        else:
            result = "\nНет доступных коллекций."
    except Exception as e:
        diag_logger.error(f"Ошибка подключения к Qdrant при получении списка коллекций: {e}")
        result = f"\nОшибка получения списка коллекций: {str(e)}"

    _collections_cache["data"] = result
    _collections_cache["timestamp"] = now
    return result


# --- Кастомный прокси для обработки MCP JSON-RPC (tools/list, tools/call) ---
from starlette.responses import Response as StarletteResponse
import inspect
import sys


async def _forward_request_to_subapp(request: Request):
    body_bytes = await request.body()
    try:
        payload = json.loads(body_bytes.decode('utf-8')) if body_bytes else {}
        diag_logger.debug(f"[MCP-DEBUG] Parsed payload: {payload}")
    except Exception as e:
        diag_logger.error(f"[MCP-DEBUG] Failed to parse JSON body: {e}")
        payload = {}

    method = payload.get('method')
    req_id = payload.get('id')
    params = payload.get('params')
    
    diag_logger.debug(f"[MCP-DEBUG] Received method: {method}, req_id: {req_id}")

    # Собираем все функции-инструменты из текущего модуля.
    # Используем sys.modules, чтобы найти app.main
    main_module = None
    for key, mod in sys.modules.items():
        if key == 'app.main':
            main_module = mod
            break
    
    if main_module is None:
        main_module = sys.modules[__name__]
    
    tools = {}
    for name, obj in vars(main_module).items():
        if callable(obj) and name.endswith('_tool'):
            tools[name] = obj

    allowed = settings.allowed_mcp_tools_set

    # --- Обработка initialize ---
    if method == "initialize":
        diag_logger.debug("[MCP-DEBUG] Handling initialize")
        result = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "Document Search MCP", "version": "1.2.0"}
            }
        }
        return StarletteResponse(content=json.dumps(result, ensure_ascii=False), media_type="application/json")

    # --- Обработка ping (если используется) ---
    if method == "ping":
        diag_logger.debug("[MCP-DEBUG] Handling ping")
        result = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        return StarletteResponse(content=json.dumps(result, ensure_ascii=False), media_type="application/json")

    # --- Обработка уведомлений (notifications) ---
    if method and method.startswith("notifications/"):
        diag_logger.debug(f"[MCP-DEBUG] Handling notification: {method}")
        # На уведомления не нужно отвечать, просто закрываем соединение
        return Response(status_code=202)  # Accepted

    # --- Обработка tools/list ---
    if method == "tools/list":
        # Однократно получаем информацию о коллекциях для всех инструментов
        collections_info = await get_collections_info()
        tools_info = []
        for tool_name, tool_fn in tools.items():
            if allowed and tool_name not in allowed:
                continue

            original_description = (tool_fn.__doc__ or '').strip()

            if tool_name == "search_documents_tool":
                description = original_description + collections_info
            else:
                description = original_description

            params_schema = None
            try:
                sig = inspect.signature(tool_fn)
                params_list = list(sig.parameters.values())
                if params_list:
                    ann = params_list[0].annotation
                    if ann is not inspect._empty:
                        try:
                            params_schema = ann.model_json_schema()
                        except (AttributeError, TypeError):
                            try:
                                params_schema = ann.schema()
                            except (AttributeError, TypeError):
                                pass
            except (TypeError, ValueError):
                pass

            tool_info = {
                "name": tool_name,
                "description": description,
                "inputSchema": params_schema or {"type": "object", "properties": {}}
            }
            tools_info.append(tool_info)

        result = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_info}}
        return StarletteResponse(content=json.dumps(result, ensure_ascii=False), media_type="application/json")

    # --- Обработка tools/call ---
    if method == "tools/call":
        diag_logger.debug("[MCP-DEBUG] Handling tools/call")
        tool_name = params.get("name") if isinstance(params, dict) else None
        tool_args = params.get("arguments", {}) if isinstance(params, dict) else {}
        if not tool_name:
            error = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Missing tool name"}}
            return StarletteResponse(content=json.dumps(error, ensure_ascii=False), status_code=400, media_type="application/json")

        if allowed and tool_name not in allowed:
            error = {"jsonrpc": "2.0", "id": req_id,
                     "error": {"code": -32000, "message": f"Tool '{tool_name}' is not allowed"}}
            return StarletteResponse(content=json.dumps(error, ensure_ascii=False), status_code=403, media_type="application/json")

        if tool_name not in tools:
            error = {"jsonrpc": "2.0", "id": req_id,
                     "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
            return StarletteResponse(content=json.dumps(error, ensure_ascii=False), status_code=404, media_type="application/json")

        tool_fn = tools[tool_name]
        try:
            sig = inspect.signature(tool_fn)
            params_list = list(sig.parameters.values())
            if params_list:
                ann = params_list[0].annotation
                if ann is not inspect._empty:
                    # Обрабатываем фильтр: если строка - парсим как JSON
                    if isinstance(tool_args, dict) and tool_args.get('filter'):
                        filter_val = tool_args['filter']
                        if isinstance(filter_val, str):
                            try:
                                tool_args['filter'] = json.loads(filter_val)
                            except (json.JSONDecodeError, TypeError):
                                pass
                    call_arg = ann(**tool_args) if isinstance(tool_args, dict) else ann(tool_args)
                else:
                    call_arg = tool_args
            else:
                call_arg = None

            if inspect.iscoroutinefunction(tool_fn):
                result_content = await tool_fn(call_arg) if call_arg is not None else await tool_fn()
            else:
                result_content = tool_fn(call_arg) if call_arg is not None else tool_fn()

            response_payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_content}]
                }
            }
            return StarletteResponse(content=json.dumps(response_payload, default=str, ensure_ascii=False),
                                     media_type="application/json")
        except ValidationError as e:
            diag_logger.error(f"Tool validation error: {e}")
            # ValidationError (ошибки валидации Pydantic) -> -32602 Invalid Request
            error_payload = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": str(e)}}
            return StarletteResponse(content=json.dumps(error_payload, ensure_ascii=False), media_type="application/json",
                                     status_code=400)
        except Exception as e:
            diag_logger.exception("Tool execution failed")
            error_payload = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(e)}}
            return StarletteResponse(content=json.dumps(error_payload, ensure_ascii=False), media_type="application/json",
                                     status_code=500)

    # --- Для остальных методов (initialize, ping, ...) передаём в mcp_app ---
    diag_logger.debug(f"[MCP-DEBUG] Forwarding to mcp_app: method={method}")
    scope = dict(request.scope)
    scope['path'] = '/mcp'

    async def receive():
        return {'type': 'http.request', 'body': body_bytes, 'more_body': False}

    response_start = {}
    body_chunks = []

    async def send(message):
        mtype = message.get('type')
        if mtype == 'http.response.start':
            response_start['status'] = message.get('status')
            raw_headers = message.get('headers', [])
            response_start['headers'] = {k.decode('latin1'): v.decode('latin1') for k, v in raw_headers}
        elif mtype == 'http.response.body':
            body_chunks.append(message.get('body', b''))

    await mcp_app(scope, receive, send)

    content = b''.join(body_chunks)
    status = response_start.get('status', 200)
    headers = response_start.get('headers', {})
    media_type = headers.get('content-type')
    return StarletteResponse(content=content, status_code=status, headers=headers, media_type=media_type)


# Регистрируем прокси-маршруты для MCP
for p in ["/mcp", "/mcp/", "/mcp/rpc", "/mcp/rpc/"]:
    app.add_api_route(p, _forward_request_to_subapp, methods=["GET", "POST", "DELETE", "OPTIONS", "HEAD"])

# Монтируем mcp_app для fallback (не используется напрямую, но нужно для initialize/ping)
app.mount("/_internal_mcp", mcp_app)


# --- Корневой эндпоинт (только один) ---
@app.get("/")
async def root():
    return {"message": "Document Vector Search Service is running", "docs": "/docs"}


# --- Lifespan event handlers (instead of deprecated on_event) ---
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup
    settings.configure_logging()
    diag_logger.info("Starting up RAG service...")
    # Регистрация инструментов через FastApiMCP больше не нужна, т.к. используется кастомный прокси.
    # Оставляем mcp_app для инициализации, если потребуется.
    diag_logger.info("MCP tools are ready via custom proxy.")
    yield
    # Shutdown (если потребуется)


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=False
    )