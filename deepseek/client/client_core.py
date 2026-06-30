"""
DeepSeek API client core

Python port of the Go DeepSeek client for interacting with the DeepSeek chat API.
Supports authentication, session management, and both streaming and non-streaming completions.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import httpx

from config.logger import get_logger
from config.store import ConfigStore
from deepseek.protocol.constants import (
    DeepSeekHost,
    DeepSeekLoginURL,
    DeepSeekCreateSessionURL,
    DeepSeekCreatePowURL,
    DeepSeekCompletionURL,
    DeepSeekContinueURL,
    DeepSeekUploadFileURL,
    DeepSeekFetchFilesURL,
    DeepSeekFetchSessionURL,
    DeepSeekDeleteSessionURL,
    DeepSeekDeleteAllSessionsURL,
    DeepSeekCompletionTargetPath,
    BaseHeaders,
    ClientVersion,
)
from .errors import (
    DeepSeekError,
    ErrRetryable,
    ErrUnauthorized,
    ErrRateLimit,
    is_token_invalid,
    is_rate_limit,
    get_error_message,
)
from .pow import PowChallenge, PowSolver, solve_pow, parse_pow_response
from .proxy import ProxyManager, get_proxy_manager

logger = get_logger("deepseek_client")


class FailureKind:
    """Error kind constants matching Go RequestFailureKind."""
    UNKNOWN = "unknown"
    DIRECT_UNAUTHORIZED = "direct_unauthorized"
    MANAGED_UNAUTHORIZED = "managed_unauthorized"


class RequestFailure(Exception):
    """Request failure exception with error kind."""
    def __init__(self, op: str, kind: str, message: str):
        super().__init__(f"{op}: {message}" if op and message else (op or message))
        self.op = op
        self.kind = kind
        self.message = message

    def __repr__(self):
        return f"RequestFailure(op={self.op!r}, kind={self.kind!r}, message={self.message!r})"


def is_managed_unauthorized(err: Exception) -> bool:
    return isinstance(err, RequestFailure) and err.kind == FailureKind.MANAGED_UNAUTHORIZED


def is_direct_unauthorized(err: Exception) -> bool:
    return isinstance(err, RequestFailure) and err.kind == FailureKind.DIRECT_UNAUTHORIZED


@dataclass
class LoginResult:
    """Result of login operation."""
    token: str = ""
    refresh_token: str = ""
    success: bool = False
    error: str = ""


@dataclass
class SessionResult:
    """Result of session creation."""
    session_id: str = ""
    success: bool = False
    error: str = ""


@dataclass
class UploadFileResult:
    """Result of file upload."""
    file_id: str = ""
    file_name: str = ""
    success: bool = False
    error: str = ""


@dataclass
class DeleteSessionResult:
    """Result of session deletion."""
    session_id: str = ""
    success: bool = False
    error: str = ""


class Client:
    """
    DeepSeek API client.

    Provides methods for:
    - Authentication (email/phone login)
    - Session management (create, delete)
    - Chat completions (streaming and non-streaming)
    - File uploads
    - PoW challenge solving
    """

    def __init__(
        self,
        store: Optional[ConfigStore] = None,
        auth_resolver: Optional[Any] = None,
    ):
        self._store = store or ConfigStore()
        self._auth_resolver = auth_resolver
        self._max_retries = 3
        self._timeout = 120.0  # seconds

        # HTTP clients (one per proxy configuration)
        self._proxy_manager = get_proxy_manager()
        self._client_lock = threading.Lock()
        self._clients: Dict[str, httpx.Client] = {}

        # PoW solver
        self._pow_solver = PowSolver()

        # Base headers
        self._base_headers = dict(BaseHeaders)

    def _get_client(self, proxy: Optional[str] = None) -> httpx.Client:
        """Get or create HTTP client for proxy configuration."""
        proxy_key = proxy or "__direct__"

        with self._client_lock:
            if proxy_key not in self._clients:
                transport, _, _ = self._proxy_manager.get_client(proxy)
                self._clients[proxy_key] = httpx.Client(
                    timeout=httpx.Timeout(self._timeout),
                    transport=transport,
                    headers=self._base_headers,
                )
            return self._clients[proxy_key]

    def _auth_headers(self, token: str) -> Dict[str, str]:
        """Build authorization headers."""
        return {
            "Authorization": f"Bearer {token}",
            **self._base_headers,
        }

    def _post_json(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Any,
        proxy: Optional[str] = None,
    ) -> tuple:
        """
        POST JSON request.

        Returns:
            (response_dict, status_code, error)
        """
        client = self._get_client(proxy)

        try:
            response = client.post(
                url,
                json=payload,
                headers=headers,
            )
            status = response.status_code

            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {}

            if status != 200:
                return data, status, get_error_message(data)

            return data, status, None

        except httpx.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return {}, 0, str(e)

    def _get_json(
        self,
        url: str,
        headers: Dict[str, str],
        proxy: Optional[str] = None,
    ) -> tuple:
        """GET JSON request."""
        client = self._get_client(proxy)

        try:
            response = client.get(url, headers=headers)
            status = response.status_code

            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {}

            return data, status, None

        except httpx.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return {}, 0, str(e)

    # === Authentication ===

    def login_email(self, email: str, password: str) -> LoginResult:
        """
        Login with email and password.

        Args:
            email: User email
            password: User password

        Returns:
            LoginResult with token on success
        """
        payload = {
            "email": email,
            "password": password,
            "device_id": "ds2api-python-proxy",
            "os": "web",
        }

        data, status, err = self._post_json(
            DeepSeekLoginURL,
            {"Content-Type": "application/json"},
            payload,
        )

        if err:
            return LoginResult(error=err)

        # Try nested response format: data.data.biz_data.user.token
        token = ""
        refresh_token = ""
        if isinstance(data, dict):
            outer_data = data.get("data", {})
            if isinstance(outer_data, dict):
                biz_data = outer_data.get("biz_data", {})
                if isinstance(biz_data, dict):
                    user = biz_data.get("user", {})
                    if isinstance(user, dict):
                        token = user.get("token", "")
                        refresh_token = user.get("refresh_token", "")
            # Fallback to flat format or biz_data directly
            if not token:
                biz_data = data.get("biz_data", {})
                if isinstance(biz_data, dict):
                    user = biz_data.get("user", {})
                    if isinstance(user, dict):
                        token = user.get("token", "")
                        refresh_token = user.get("refresh_token", "")
            if not token:
                token = data.get("token", "")
                refresh_token = data.get("refresh_token", "")

        if not token:
            return LoginResult(error="No token in response")

        return LoginResult(
            token=token,
            refresh_token=refresh_token,
            success=True,
        )

    def login_phone(self, phone: str, code: str) -> LoginResult:
        """
        Login with phone and verification code.

        Args:
            phone: Phone number
            code: Verification code

        Returns:
            LoginResult with token on success
        """
        payload = {
            "phone": phone,
            "code": code,
        }

        data, status, err = self._post_json(
            DeepSeekLoginURL,
            {"Content-Type": "application/json"},
            payload,
        )

        if err:
            return LoginResult(error=err)

        token = data.get("token", "")
        refresh_token = data.get("refresh_token", "")

        if not token:
            return LoginResult(error="No token in response")

        return LoginResult(
            token=token,
            refresh_token=refresh_token,
            success=True,
        )

    def login(self, account: Dict[str, Any]) -> str:
        """
        Login to DeepSeek using account credentials.

        Args:
            account: Account dict with email/phone/password

        Returns:
            Authentication token

        Raises:
            DeepSeekError: If login fails
        """
        email = account.get("email", "")
        phone = account.get("phone", "")
        password = account.get("password", "")

        if email and password:
            result = self.login_email(email, password)
            if result.success:
                return result.token
            raise DeepSeekError(f"Login failed: {result.error}")

        if phone:
            # Phone login requires verification code
            raise DeepSeekError("Phone login requires verification code")

        raise DeepSeekError("No login credentials provided")

    def refresh_token(self, refresh_token: str) -> Optional[str]:
        """Refresh authentication token."""
        # Implementation depends on DeepSeek API
        return None

    # === Session Management ===

    def create_session(
        self,
        token: str,
        max_attempts: int = 3,
    ) -> str:
        """
        Create a new chat session.

        Args:
            token: Authentication token
            max_attempts: Maximum retry attempts

        Returns:
            Session ID

        Raises:
            DeepSeekError: If session creation fails
        """
        payload = {}

        for attempt in range(max_attempts):
            data, status, err = self._post_json(
                DeepSeekCreateSessionURL,
                self._auth_headers(token),
                payload,
            )

            if err:
                logger.warning(f"Create session attempt {attempt + 1} failed: {err}")
                continue

            code = data.get("code", 0)
            session_id = ""
            if code == 0:
                # Try nested format first: data.data.biz_data.id
                outer_data = data.get("data", {})
                if isinstance(outer_data, dict):
                    biz_data = outer_data.get("biz_data", {})
                    if isinstance(biz_data, dict):
                        session_id = biz_data.get("id", "")
                # Fallback to flat format
                if not session_id:
                    session_id = data.get("session_id", "")
                    if not session_id:
                        session_id = data.get("id", "")
                if session_id:
                    return session_id

            logger.warning(f"Create session attempt {attempt + 1}: code={code}")

        raise DeepSeekError("Failed to create session after retries")

    def delete_session(
        self,
        token: str,
        session_id: str,
    ) -> DeleteSessionResult:
        """Delete a chat session."""
        payload = {"chat_session_id": session_id}

        data, status, err = self._post_json(
            DeepSeekDeleteSessionURL,
            self._auth_headers(token),
            payload,
        )

        if err:
            return DeleteSessionResult(
                session_id=session_id,
                success=False,
                error=err,
            )

        code = data.get("code", 0)
        if status == 200 and code == 0:
            return DeleteSessionResult(
                session_id=session_id,
                success=True,
            )

        return DeleteSessionResult(
            session_id=session_id,
            success=False,
            error=get_error_message(data),
        )

    def delete_all_sessions(self, token: str) -> bool:
        """Delete all chat sessions."""
        payload = {}

        data, status, err = self._post_json(
            DeepSeekDeleteAllSessionsURL,
            self._auth_headers(token),
            payload,
        )

        if err:
            return False

        code = data.get("code", 0)
        return status == 200 and code == 0

    # === PoW ===

    def get_pow(
        self,
        token: str,
        max_attempts: int = 3,
    ) -> str:
        """
        Get and solve a PoW challenge using DeepSeekPowSolver (Node.js + WASM).

        1. Fetch challenge via our HTTP client
        2. Solve via DeepSeekPowSolver.SleviS() (Node.js + WASM)
        3. Format via DeepSeekPowSolver.create_Fis()

        Returns:
            Base64-encoded PoW response string, or empty string on failure
        """
        for attempt in range(max_attempts):
            data, status, err = self._post_json(
                DeepSeekCreatePowURL,
                self._auth_headers(token),
                {"target_path": DeepSeekCompletionTargetPath},
            )
            if err or status != 200:
                logger.warning(f"PoW fetch attempt {attempt + 1} failed: {err}")
                continue

            code = data.get("code", 0)
            if code != 0:
                logger.warning(f"PoW fetch attempt {attempt + 1}: code={code}")
                continue

            # Extract challenge dict
            outer_data = data.get("data", {})
            biz_data = outer_data.get("biz_data", {}) if isinstance(outer_data, dict) else {}
            challenge = biz_data.get("challenge", {}) if isinstance(biz_data, dict) else {}

            if not challenge:
                logger.warning(f"PoW: no challenge data in response")
                continue

            # Solve via DeepSeekPowSolver
            try:
                from deepseekpowsolver import DeepSeekPowSolver as DSPSolver
                import tempfile as _tmp, subprocess as _sp
                solver = DSPSolver(token=token)
                wasm_fixed = str(solver.wasm_path.absolute()).replace('\\', '/')
                
                # Build and run JS solver directly (avoiding the path escaping bug in SleviS)
                _js = f"""const {{ readFileSync }} = require('fs');
const ch = "{challenge['challenge']}";
const salt = "{challenge['salt']}";
const diff = {challenge['difficulty']};
const exp = {challenge['expire_at']};
async function solve() {{
    const wb = readFileSync('{wasm_fixed}');
    const {{ instance }} = await WebAssembly.instantiate(wb, {{ wbg: {{}} }});
    const w = instance.exports;
    const te = new TextEncoder();
    const es = (s) => {{ const b = te.encode(s); const p = w.__wbindgen_export_0(b.length, 1); new Uint8Array(w.memory.buffer).set(b, p); return {{ ptr: p, len: b.length }}; }};
    const a = es(ch), b = es(salt+'_'+exp+'_');
    const rp = w.__wbindgen_add_to_stack_pointer(-16);
    w.wasm_solve(rp, a.ptr, a.len, b.ptr, b.len, diff);
    const dv = new DataView(w.memory.buffer);
    const st = dv.getInt32(rp + 0, true);
    const v = dv.getFloat64(rp + 8, true);
    w.__wbindgen_add_to_stack_pointer(16);
    if (st !== 0) console.log(Math.round(v).toString());
}}
solve().catch(e => {{}});"""
                with _tmp.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                    f.write(_js)
                    js_path = f.name
                result = _sp.run(['node', js_path], capture_output=True, text=True, timeout=30)
                import os as _os; _os.unlink(js_path)
                answer_str = result.stdout.strip()
                if answer_str and answer_str.isdigit():
                    answer = int(answer_str)
                    if answer > 0:
                        pow_header = solver.create_Fis(challenge, answer)
                        if pow_header:
                            logger.info(f"PoW solved (attempt {attempt + 1}, answer={answer})")
                            return pow_header
                logger.warning(f"PoW solve attempt {attempt + 1}: no valid answer (stdout={result.stdout[:50]}, stderr={result.stderr[:100]})")
            except Exception as e:
                logger.warning(f"DeepSeekPowSolver error: {e}")

        logger.warning("PoW failed after all attempts, continuing without PoW")
        return ""

    def preload_pow(self) -> None:
        """Preload and cache PoW challenge for faster first request."""
        # In Python, we don't need explicit preload
        # The solver initializes on first use
        pass

    # === File Upload ===

    def upload_file(
        self,
        token: str,
        file_data: bytes,
        filename: str,
        max_attempts: int = 3,
    ) -> UploadFileResult:
        """
        Upload a file to DeepSeek.

        Args:
            token: Authentication token
            file_data: File content as bytes
            filename: File name
            max_attempts: Maximum retry attempts

        Returns:
            UploadFileResult with file_id on success
        """
        client = self._get_client()

        for attempt in range(max_attempts):
            try:
                files = {"file": (filename, file_data)}
                data = {"purpose": "chat"}

                response = client.post(
                    DeepSeekUploadFileURL,
                    files=files,
                    data=data,
                    headers=self._auth_headers(token),
                )

                if response.status_code == 200:
                    result = response.json()
                    file_id = result.get("id", "")
                    if file_id:
                        return UploadFileResult(
                            file_id=file_id,
                            file_name=filename,
                            success=True,
                        )

                logger.warning(f"Upload file attempt {attempt + 1} failed")

            except httpx.HTTPError as e:
                logger.warning(f"Upload file attempt {attempt + 1} error: {e}")

        return UploadFileResult(
            file_name=filename,
            success=False,
            error="Upload failed after retries",
        )

    def get_file_status(
        self,
        token: str,
        file_id: str,
    ) -> Dict[str, Any]:
        """Get file status."""
        url = f"{DeepSeekFetchFilesURL}/{file_id}"
        data, status, err = self._get_json(
            url,
            self._auth_headers(token),
        )

        if err or status != 200:
            return {"error": err or "Failed to get file status"}

        return data

    # === Proxy Request ===

    def proxy_request(
        self,
        method: str,
        path: str,
        token: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Any] = None,
        proxy: Optional[str] = None,
    ) -> tuple:
        """
        Make a proxy request to DeepSeek API.

        Args:
            method: HTTP method
            path: API path
            token: Authentication token
            headers: Additional headers
            json_data: JSON body
            proxy: Optional proxy URL

        Returns:
            (response_dict, status_code, error)
        """
        url = f"https://{DeepSeekHost}{path}"

        req_headers = self._auth_headers(token)
        if headers:
            req_headers.update(headers)

        if json_data is not None:
            return self._post_json(url, req_headers, json_data, proxy)
        else:
            return self._get_json(url, req_headers, proxy)

    def completion(
        self,
        token: str,
        session_id: str,
        payload: dict,
        pow_token: Optional[str] = None,
        stream: bool = False,
        max_attempts: int = 3,
    ):
        """
        Send a chat completion request (handler-compatible bridge).

        Handles SSE-formatted responses from DeepSeek API.

        Returns:
            (status_code, response) tuple where response is JSON dict or str on error
        """
        from .client_completion import call_completion, _nonstream_completion, _stream_completion

        try:
            http_client = self._get_client()
            headers = self._auth_headers(token)
            headers["Content-Type"] = "application/json"
            if pow_token:
                headers["X-Ds-Pow-Response"] = pow_token

            if stream:
                response = call_completion(
                    self, token, session_id, payload, pow_token, max_attempts
                )
                return (response.status_code, response)
            else:
                # Use _nonstream_completion which properly parses SSE
                result = _nonstream_completion(
                    http_client,
                    DeepSeekCompletionURL,
                    headers,
                    payload,
                    payload.get("thinking_enabled", False),
                )
                if result.is_error:
                    return (500, result.error)
                return (200, {
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.content,
                            "reasoning_content": result.thinking,
                        },
                        "finish_reason": result.finish_reason,
                    }],
                    "usage": result.usage,
                })
        except Exception as e:
            return (500, str(e))

    def close(self) -> None:
        """Close all HTTP clients."""
        with self._client_lock:
            for client in self._clients.values():
                client.close()
            self._clients.clear()
