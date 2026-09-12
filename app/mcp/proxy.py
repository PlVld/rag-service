# app/mcp/proxy.py
# Кастомный прокси для обработки MCP JSON-RPC (initialize, ping, tools/list, tools/call).
# Инструменты берутся из явного реестра TOOLS (app/mcp/tools.py).

import inspect
import json
import logging

from fastapi import FastAPI, Request
from pydantic import ValidationError
from starlette.responses import Response as StarletteResponse

from app.core.config import settings
from app.mcp.tools import TOOLS, get_collections_info

diag_logger = logging.getLogger("rag_service.diagnostics")

# Fallback subapp для методов, которые прокси не обрабатывает сам
mcp_app = FastAPI(title="MCP subapp")

MCP_HTTP_METHODS = ["GET", "POST", "DELETE", "OPTIONS", "HEAD"]
MCP_PATHS = ["/mcp", "/mcp/", "/mcp/rpc", "/mcp/rpc/"]


def register_mcp_routes(app) -> None:
    """Регистрирует MCP-прокси на всех алиасах пути одним маршрутом на каждый."""
    for path in MCP_PATHS:
        app.add_api_route(
            path,
            _forward_request_to_subapp,
            methods=MCP_HTTP_METHODS,
            include_in_schema=False,
        )


def _jsonrpc_response(payload, status_code: int = 200) -> StarletteResponse:
    return StarletteResponse(
        content=json.dumps(payload, default=str, ensure_ascii=False),
        status_code=status_code,
        media_type="application/json",
    )


def _build_input_model(tool_fn, tool_args):
    """Создаёт аргумент инструмента из Pydantic-аннотации первого параметра."""
    sig = inspect.signature(tool_fn)
    params_list = list(sig.parameters.values())
    if not params_list:
        return tool_args
    ann = params_list[0].annotation
    if ann is inspect._empty:
        return tool_args
    # Обрабатываем фильтр: если строка - парсим как JSON
    if isinstance(tool_args, dict) and tool_args.get('filter'):
        filter_val = tool_args['filter']
        if isinstance(filter_val, str):
            try:
                tool_args['filter'] = json.loads(filter_val)
            except (json.JSONDecodeError, TypeError):
                pass
    return ann(**tool_args) if isinstance(tool_args, dict) else ann(tool_args)


def _input_schema(tool_fn) -> dict:
    """Достаёт JSON-схему входных параметров инструмента из аннотации."""
    try:
        sig = inspect.signature(tool_fn)
        params_list = list(sig.parameters.values())
        if params_list:
            ann = params_list[0].annotation
            if ann is not inspect._empty:
                try:
                    return ann.model_json_schema()
                except (AttributeError, TypeError):
                    try:
                        return ann.schema()
                    except (AttributeError, TypeError):
                        pass
    except (TypeError, ValueError):
        pass
    return {"type": "object", "properties": {}}


async def _handle_tools_list(req_id) -> StarletteResponse:
    # Однократно получаем информацию о коллекциях для всех инструментов
    collections_info = await get_collections_info()
    allowed = settings.allowed_mcp_tools_set

    tools_info = []
    for tool_name, tool_fn in TOOLS.items():
        if allowed and tool_name not in allowed:
            continue
        description = (tool_fn.__doc__ or '').strip()
        if tool_name == "search_documents_tool":
            description += collections_info
        tools_info.append({
            "name": tool_name,
            "description": description,
            "inputSchema": _input_schema(tool_fn),
        })

    return _jsonrpc_response({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_info}})


async def _handle_tools_call(req_id, params) -> StarletteResponse:
    tool_name = params.get("name") if isinstance(params, dict) else None
    tool_args = params.get("arguments", {}) if isinstance(params, dict) else {}
    allowed = settings.allowed_mcp_tools_set

    if not tool_name:
        return _jsonrpc_response(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Missing tool name"}},
            status_code=400,
        )
    if allowed and tool_name not in allowed:
        return _jsonrpc_response(
            {"jsonrpc": "2.0", "id": req_id,
             "error": {"code": -32000, "message": f"Tool '{tool_name}' is not allowed"}},
            status_code=403,
        )
    if tool_name not in TOOLS:
        return _jsonrpc_response(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}},
            status_code=404,
        )

    tool_fn = TOOLS[tool_name]
    try:
        call_arg = _build_input_model(tool_fn, tool_args)
        if inspect.iscoroutinefunction(tool_fn):
            result_content = await tool_fn(call_arg) if call_arg is not None else await tool_fn()
        else:
            result_content = tool_fn(call_arg) if call_arg is not None else tool_fn()

        return _jsonrpc_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": result_content}]},
        })
    except ValidationError as e:
        diag_logger.error(f"Tool validation error: {e}")
        # ValidationError (ошибки валидации Pydantic) -> -32602 Invalid Request
        return _jsonrpc_response(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": str(e)}},
            status_code=400,
        )
    except Exception as e:
        diag_logger.exception("Tool execution failed")
        return _jsonrpc_response(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(e)}},
            status_code=500,
        )


async def _forward_request_to_subapp(request: Request):
    body_bytes = await request.body()
    try:
        payload = json.loads(body_bytes.decode('utf-8')) if body_bytes else {}
    except Exception as e:
        diag_logger.error(f"[MCP-DEBUG] Failed to parse JSON body: {e}")
        payload = {}

    method = payload.get('method')
    req_id = payload.get('id')
    params = payload.get('params')

    diag_logger.debug(f"[MCP-DEBUG] Received method: {method}, req_id: {req_id}")

    # --- Обработка initialize ---
    if method == "initialize":
        result = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "Document Search MCP", "version": "1.2.0"}
            }
        }
        return _jsonrpc_response(result)

    # --- Обработка ping (если используется) ---
    if method == "ping":
        return _jsonrpc_response({"jsonrpc": "2.0", "id": req_id, "result": {}})

    # --- Обработка уведомлений (notifications) ---
    if method and method.startswith("notifications/"):
        # На уведомления не нужно отвечать, просто закрываем соединение
        return StarletteResponse(status_code=202)  # Accepted

    # --- Обработка tools/list ---
    if method == "tools/list":
        return await _handle_tools_list(req_id)

    # --- Обработка tools/call ---
    if method == "tools/call":
        return await _handle_tools_call(req_id, params)

    # --- Для остальных методов передаём в mcp_app ---
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
