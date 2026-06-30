"""ASGI Server Router - Main entry point for the ds2api server."""
import contextlib
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from starlette.applications import Starlette
from starlette.responses import (
    JSONResponse,
    Response,
    StreamingResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
)
from starlette.routing import Route, Mount, Match
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

# Local imports
from config.store import ConfigStore, LoadStoreWithError
from config.logger import Logger, setup_logging
from account.pool_core import Pool
from auth.request import RequestAuth, AuthResolver
from auth.admin import AdminAuth
from deepseek.client.client_core import Client
from chathistory.store import ChatHistoryStore
from httpapi.openai.chat.handler_chat import Handler as ChatHandler


# ─── Request ID Middleware ───────────────────────────────────────────────────


class RequestIDMiddleware:
    """Add X-Request-ID header to each request."""

    def __init__(self, app: ASGIApp):
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        request_id = str(uuid.uuid4())
        scope["state"]["request_id"] = request_id

        async def send_with_id(message: Dict[str, Any]):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-request-id"] = request_id.encode()
                message = {**message, "headers": list(headers.items())}
            await send(message)

        await self._app(scope, receive, send_with_id)


class RealIPMiddleware:
    """Extract real IP from X-Forwarded-For header."""

    def __init__(self, app: ASGIApp):
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        headers = dict(scope.get("headers", []))
        forwarded = headers.get(b"x-forwarded-for", b"").decode()
        if forwarded:
            scope["client"] = (forwarded.split(",")[0].strip(), 0)
        await self._app(scope, receive, send)


class RecoverMiddleware:
    """Catch exceptions and return 500."""

    def __init__(self, app: ASGIApp):
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        sent_start = False
        try:
            await self._app(scope, receive, send)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            traceback.print_exc()
            await send({
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    (b"content-type", b"application/json"),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({
                    "error": {
                        "message": str(e),
                        "code": "internal_error",
                        "traceback": tb,
                    }
                }).encode(),
            })


# ─── App ─────────────────────────────────────────────────────────────────────


class App:
    """Main application."""

    def __init__(self, config_path: Optional[str] = None):
        self._config_path = config_path or os.environ.get("CONFIG_PATH", "config.json")
        self._store: Optional[ConfigStore] = None
        self._pool: Optional[Pool] = None
        self._resolver: Optional[AuthResolver] = None
        self._ds_client: Optional[Client] = None
        self._chat_history: Optional[ChatHistoryStore] = None
        self._chat_handler: Optional[ChatHandler] = None
        self._admin_auth: Optional[AdminAuth] = None
        self._routes = self._build_routes()
        self._starlette = self._build_starlette()

    # ─── Setup ────────────────────────────────────────────────────────────

    def setup(self):
        """Initialize all components."""
        setup_logging()

        # Load config
        os.environ["DS2API_CONFIG_PATH"] = self._config_path
        self._store = LoadStoreWithError()
        Logger.Info("[app] config loaded", path=self._config_path)

        # Account pool
        self._pool = Pool(self._store)

        # Auth resolver
        self._resolver = AuthResolver(self._store, self._pool)

        # DeepSeek client
        self._ds_client = Client(self._store, self._resolver)

        # Wire up login function for auto-token retrieval
        self._resolver.login_func = self._ds_client.login

        # Preload PoW
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._ds_client.preload_pow())
            Logger.Info("[app] PoW solver ready")
        except Exception as e:
            Logger.Warn("[app] PoW preload failed", error=str(e))

        # Chat history
        history_path = self._store.chat_history_path()
        if history_path:
            self._chat_history = ChatHistoryStore(history_path)
            Logger.Info("[app] chat history ready", path=history_path)

        # Handlers
        self._chat_handler = ChatHandler(
            store=self._store,
            auth_resolver=self._resolver,
            ds_client=self._ds_client,
            chat_history=self._chat_history,
        )

        self._admin_auth = AdminAuth(self._store, self._pool, self._ds_client)

        Logger.Info("[app] setup complete")

    # ─── Routes ────────────────────────────────────────────────────────────

    def _build_routes(self) -> List[Route]:
        return [
            # Health check
            Route("/health", self._handle_health, methods=["GET"]),

            # OpenAI-compatible /v1/ endpoints
            Route("/v1/chat/completions", self._handle_chat_completions, methods=["POST"]),
            Route("/v1/models", self._handle_list_models, methods=["GET"]),
            Route("/v1/models/{model}", self._handle_get_model, methods=["GET"]),

            # Chat history
            Route("/v1/history/list", self._handle_history_list, methods=["GET"]),
            Route("/v1/history/delete", self._handle_history_delete, methods=["GET", "POST"]),

            # Admin
            Route("/v1/admin/login", self._handle_admin_login, methods=["POST"]),
            Route("/v1/admin/config", self._handle_admin_config, methods=["GET", "POST"]),

            # Pool stats
            Route("/admin/pool/stats", self._handle_pool_stats, methods=["GET"]),

            # Root / favicon
            Route("/", self._handle_root, methods=["GET"]),
            Route("/favicon.ico", self._handle_favicon, methods=["GET"]),
        ]

    @contextlib.asynccontextmanager
    async def _lifespan(self, app):
        if self._store is None:
            self.setup()
        yield

    def _build_starlette(self) -> Starlette:
        return Starlette(
            debug=os.environ.get("DEBUG", "").lower() in ("1", "true"),
            routes=self._routes,
            middleware=[
                Middleware(RecoverMiddleware),
                Middleware(RealIPMiddleware),
                Middleware(RequestIDMiddleware),
                Middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["*"],
                    allow_headers=["*"],
                    allow_credentials=True,
                ),
                Middleware(GZipMiddleware, minimum_size=1024),
            ],
            lifespan=self._lifespan,
        )

    # ─── Request Handlers ──────────────────────────────────────────────────

    async def _handle_health(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "time": int(time.time())})

    async def _handle_root(self, request: Request) -> JSONResponse:
        return JSONResponse({
            "name": "ds2api",
            "version": "2.0.4",
            "description": "DeepSeek API proxy with account pooling",
        })

    async def _handle_favicon(self, request: Request) -> Response:
        return Response(content=b"", media_type="image/x-icon")

    async def _handle_chat_completions(self, request: Request) -> Union[JSONResponse, StreamingResponse]:
        """POST /v1/chat/completions"""
        self._ensure_setup()

        # Read body
        try:
            body = await request.body()
            if len(body) > 10 * 1024 * 1024:
                return JSONResponse(
                    {"error": {"message": "Request too large", "code": "request_too_large"}},
                    status_code=413,
                )
            req = json.loads(body)
        except json.JSONDecodeError:
            return JSONResponse(
                {"error": {"message": "Invalid JSON", "code": "invalid_request"}},
                status_code=400,
            )

        # External API Key validation
        api_keys = self._store.data.get("api_keys", [])
        external_key_valid = False
        if api_keys:
            auth_header = request.headers.get("authorization", "")
            key = auth_header.replace("Bearer ", "").replace("bearer ", "")
            if key not in api_keys:
                return JSONResponse({"error": {"message": "Invalid API key", "type": "authentication_error"}}, status_code=401)
            external_key_valid = True

        # Auth (external key validated above, still auto-auth for DeepSeek token)
        try:
            auth_obj = self._resolver.determine(request, skip_deepseek=False)
            auth_result = (200, auth_obj, "")
        except ValueError as e:
            auth_result = (401, None, str(e))

        # Handle
        result = self._chat_handler.chat_completions(req, auth_result)

        if result.get("type") == "error":
            return JSONResponse(result.get("error", {}), status_code=result.get("status", 500))

        if result.get("type") == "non_stream":
            return JSONResponse(result["body"])

        if result.get("type") == "stream":
            streamer = result["streamer"]
            return StreamingResponse(
                self._stream_response(streamer()),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return JSONResponse(
            {"error": {"message": "Unknown response type", "code": "internal_error"}},
            status_code=500,
        )

    async def _stream_response(self, generator):
        """Wrap a sync generator as async."""
        for chunk in generator:
            if isinstance(chunk, str):
                yield chunk.encode("utf-8")
            else:
                yield chunk

    async def _handle_list_models(self, request: Request) -> JSONResponse:
        """GET /v1/models"""
        self._ensure_setup()
        models = self._store.models()
        return JSONResponse({
            "object": "list",
            "data": [
                {
                    "id": m.get("id", m.get("model", "")),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "deepseek",
                }
                for m in models
            ],
        })

    async def _handle_get_model(self, request: Request, model: str) -> JSONResponse:
        """GET /v1/models/{model}"""
        self._ensure_setup()
        return JSONResponse({
            "id": model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "deepseek",
        })

    async def _handle_history_list(self, request: Request) -> JSONResponse:
        """GET /v1/history/list"""
        self._ensure_setup()
        if not self._chat_history:
            return JSONResponse({"error": {"message": "Chat history unavailable"}}, status_code=503)

        try:
            revision = int(request.query_params.get("revision", 0))
            etag = request.headers.get("if-none-match", "")

            if revision > 0:
                expected_etag = f'W/"chat-history-list-{revision}"'
                if etag == expected_etag:
                    return JSONResponse({"status": "not_modified"}, status_code=304)

            file = self._chat_history.snapshot()
            return JSONResponse({
                "object": "list",
                "data": [item.to_dict() for item in file.items],
                "revision": file.revision,
            })
        except Exception as e:
            return JSONResponse({"error": {"message": str(e)}}, status_code=500)

    async def _handle_history_delete(self, request: Request) -> JSONResponse:
        """GET/POST /v1/history/delete"""
        self._ensure_setup()
        if not self._chat_history:
            return JSONResponse({"error": {"message": "Chat history unavailable"}}, status_code=503)

        id_param = request.query_params.get("id", "") or (await request.json()).get("id", "")
        if not id_param:
            return JSONResponse({"error": {"message": "id required"}}, status_code=400)

        try:
            self._chat_history.delete(id_param)
            return JSONResponse({"status": "ok"})
        except Exception as e:
            return JSONResponse({"error": {"message": str(e)}}, status_code=500)

    async def _handle_admin_login(self, request: Request) -> JSONResponse:
        """POST /v1/admin/login"""
        self._ensure_setup()
        try:
            body = await request.json()
            username = body.get("username", "")
            password = body.get("password", "")
        except:
            return JSONResponse({"error": {"message": "Invalid request"}}, status_code=400)

        result = self._admin_auth.login(username, password)
        if result.get("error"):
            return JSONResponse(result, status_code=401)
        return JSONResponse(result)

    async def _handle_admin_config(self, request: Request) -> JSONResponse:
        """GET/POST /v1/admin/config"""
        self._ensure_setup()

        # Check admin auth
        admin_auth = request.headers.get("authorization", "")
        if not self._admin_auth.check(auth_header(admin_auth)):
            return JSONResponse({"error": {"message": "Unauthorized"}}, status_code=401)

        if request.method == "GET":
            return JSONResponse(self._store.get_public_config())

        # POST - update config
        try:
            body = await request.json()
            self._store.update_config(body)
            return JSONResponse({"status": "ok"})
        except Exception as e:
            return JSONResponse({"error": {"message": str(e)}}, status_code=400)

    async def _handle_pool_stats(self, request: Request) -> JSONResponse:
        """GET /admin/pool/stats"""
        self._ensure_setup()
        try:
            stats = self._pool.snapshot()
            return JSONResponse(stats)
        except Exception as e:
            return JSONResponse({"error": {"message": str(e)}}, status_code=500)

    # ─── Helpers ───────────────────────────────────────────────────────────

    def _ensure_setup(self):
        if self._store is None:
            self.setup()

    # ─── ASGI App ──────────────────────────────────────────────────────────

    @property
    def asgi_app(self) -> ASGIApp:
        return self._starlette


# ─── Singleton ────────────────────────────────────────────────────────────────


_app_instance: Optional[App] = None


def get_app(config_path: Optional[str] = None) -> ASGIApp:
    """Get or create the ASGI app instance."""
    global _app_instance
    if config_path is None:
        config_path = os.environ.get("DS2API_CONFIG_PATH")
    if _app_instance is None:
        _app_instance = App(config_path)
    return _app_instance.asgi_app


# ─── Utilities ───────────────────────────────────────────────────────────────


def auth_header(value: str) -> Optional[str]:
    if not value:
        return None
    parts = value.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]
