"""OpenAI Chat Completions Handler."""
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from config.store import ConfigStore
    from auth import RequestAuth, AuthResolver
    from deepseek.client import Client
    from chathistory.store import ChatHistoryStore


MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB


class Handler:
    """Chat Completions HTTP Handler."""

    def __init__(
        self,
        store: 'ConfigStore',
        auth_resolver: 'AuthResolver',
        ds_client: 'Client',
        chat_history: Optional['ChatHistoryStore'] = None,
    ):
        self._store = store
        self._auth = auth_resolver
        self._ds = ds_client
        self._chat_history = chat_history

    # ─── Main Entry ─────────────────────────────────────────────────────────

    def chat_completions(
        self,
        request: Dict[str, Any],
        auth_result: Tuple[int, Any, str],
    ) -> Dict[str, Any]:
        """
        Handle POST /v1/chat/completions.
        
        Returns a dict with either:
        - {"type": "non_stream", "body": {...}}
        - {"type": "stream", "streamer": <callable>}
        """
        http_status, auth_obj, error_detail = auth_result

        if http_status != 200:
            return {
                "type": "error",
                "status": http_status,
                "error": {"message": error_detail, "code": "authentication_error"},
            }

        a: 'RequestAuth' = auth_obj

        # Parse and normalize request
        try:
            std_req = _normalize_openai_request(request)
        except Exception as e:
            return {
                "type": "error",
                "status": 400,
                "error": {"message": str(e), "code": "invalid_request"},
            }

        # Apply model defaults
        std_req = _apply_model_defaults(std_req, self._store)

        if not std_req.get("stream", False):
            return self._handle_non_stream(std_req, a)

        return self._handle_stream(std_req, a)

    # ─── Non-Stream ────────────────────────────────────────────────────────

    def _handle_non_stream(
        self,
        std_req: Dict[str, Any],
        auth: 'RequestAuth',
    ) -> Dict[str, Any]:
        """Handle non-streaming completion."""
        max_attempts = 3
        session_id = ""

        try:
            # Create session
            session_id = self._ds.create_session(auth.deepseek_token, max_attempts)

            # Get PoW (skip if fails)
            pow_token = self._ds.get_pow(auth.deepseek_token, 1) or ""

            # Build payload
            payload = _build_payload(std_req, session_id)

            # Execute
            status, resp = self._ds.completion(
                token=auth.deepseek_token,
                session_id=session_id,
                payload=payload,
                pow_token=pow_token,
                stream=False,
            )

            if status != 200:
                return {
                    "type": "error",
                    "status": status,
                    "error": {"message": str(resp), "code": "api_error"},
                }

            # Parse response
            turn = _parse_non_stream_response(resp, std_req)

            # Build response
            resp_body = _build_chat_completion_response(
                session_id or _make_id(),
                std_req.get("model", "deepseek-chat"),
                turn,
                std_req.get("tools"),
            )

            return {"type": "non_stream", "body": resp_body}

        except Exception as e:
            return {
                "type": "error",
                "status": 500,
                "error": {"message": str(e), "code": "internal_error"},
            }

    # ─── Stream ────────────────────────────────────────────────────────────

    def _handle_stream(
        self,
        std_req: Dict[str, Any],
        auth: 'RequestAuth',
    ) -> Dict[str, Any]:
        """Handle streaming completion. Returns a generator callable."""
        max_attempts = 3

        def stream_generator():
            session_id = ""
            try:
                session_id = self._ds.create_session(auth.deepseek_token, max_attempts)
                pow_token = self._ds.get_pow(auth.deepseek_token, 1) or ""
                payload = _build_payload(std_req, session_id)

                status, resp = self._ds.completion(
                    token=auth.deepseek_token,
                    session_id=session_id,
                    payload=payload,
                    pow_token=pow_token,
                    stream=True,
                )

                if status != 200:
                    yield _make_error_chunk(f"HTTP {status}")
                    return

                completion_id = _make_id()
                created = int(time.time())
                model = std_req.get("model", "deepseek-chat")
                thinking = std_req.get("thinking", False)

                yield from _stream_sse(resp, completion_id, created, model, thinking)

            except Exception as e:
                yield _make_error_chunk(str(e))

        return {"type": "stream", "streamer": stream_generator}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _normalize_openai_request(req: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize OpenAI chat request to standard format."""
    model = req.get("model", "deepseek-chat")
    messages = req.get("messages", [])
    stream = req.get("stream", False)
    temperature = req.get("temperature", 1.0)
    top_p = req.get("top_p", 1.0)
    max_tokens = req.get("max_tokens")
    tools = req.get("tools")
    tool_choice = req.get("tool_choice", "auto")
    thinking = req.get("thinking", False)
    search = req.get("search_enabled", False)

    # Normalize messages
    normalized_messages = _normalize_messages(messages)

    return {
        "model": model,
        "messages": normalized_messages,
        "stream": stream,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "tools": tools,
        "tool_choice": tool_choice,
        "thinking": thinking,
        "search": search,
    }


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize message format."""
    normalized = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Skip system messages with only role
        if role == "system" and not content:
            continue

        normalized_msg: Dict[str, Any] = {"role": role, "content": content}

        # Handle tool calls
        if msg.get("tool_calls"):
            normalized_msg["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            normalized_msg["tool_call_id"] = msg["tool_call_id"]

        normalized.append(normalized_msg)

    return normalized


def _apply_model_defaults(req: Dict[str, Any], store: Any) -> Dict[str, Any]:
    """Apply model-specific defaults from config."""
    model = req.get("model", "deepseek-chat")

    # Resolve aliases
    resolved = store.resolve_model(model)

    # Get model config from store
    cfg = store.get_model_config(resolved)
    if cfg is None:
        # Unknown model: use reasonable defaults
        cfg = {"thinking": True, "search": False, "vision": False}

    # Model-specific forced settings (nothinking / others)
    is_nothink = "-nothinking" in resolved.lower()
    is_search = "-search" in resolved.lower()
    is_vision = cfg.get("vision", False)

    # thinking: nothinking models force it off, otherwise use model's default
    if is_nothink:
        req_thinking = False
    elif "thinking" in req and req["thinking"] is not None:
        req_thinking = bool(req["thinking"])
    else:
        req_thinking = cfg.get("thinking", True)

    # search: models with -search in name default to True
    if "search" in req and req["search"] is not None:
        req_search = bool(req["search"])
    else:
        req_search = is_search or cfg.get("search", False)

    req = dict(req)
    req["model"] = resolved
    req["model_type"] = _get_model_type(resolved, is_vision)
    import sys; sys.stderr.write(f"[DEBUG] model={resolved} is_vision={is_vision} model_type={_get_model_type(resolved, is_vision)}\n")
    req["thinking"] = req_thinking
    req["search"] = req_search
    return req


def _build_payload(std_req: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Build DeepSeek completion payload."""
    is_vision = std_req.get("model_type") == "vision"

    if is_vision:
        # Vision: use messages format with extracted content blocks
        messages = _extract_vision_messages(std_req.get("messages", []))
        # Extract prompt text for backward compatibility
        prompt_text = _build_prompt_from_messages(std_req.get("messages", []))
        return {
            "chat_session_id": session_id,
            "model": std_req.get("model", "deepseek-v4-vision"),
            "parent_message_id": None,
            "ref_file_ids": [],
            "prompt": prompt_text,
            "messages": messages,
            "max_tokens": std_req.get("max_tokens", 8192),
            "thinking_enabled": std_req.get("thinking", False),
            "search_enabled": std_req.get("search", False),
            "stream": std_req.get("stream", False),
            "temperature": std_req.get("temperature", 1.0),
            "top_p": std_req.get("top_p", 1.0),
        }

    # Text models: use prompt format
    return {
        "chat_session_id": session_id,
        "model_type": std_req.get("model_type", "default"),
        "parent_message_id": None,
        "prompt": _build_prompt_from_messages(std_req.get("messages", [])),
        "ref_file_ids": [],
        "thinking_enabled": std_req.get("thinking", False),
        "search_enabled": std_req.get("search", False),
        "stream": std_req.get("stream", False),
        "temperature": std_req.get("temperature", 1.0),
        "top_p": std_req.get("top_p", 1.0),
    }

def _get_model_type(model: str, is_vision: bool = False) -> str:
    """Get DeepSeek model type."""
    if is_vision:
        return "vision"
    if "coder" in model.lower():
        return "coder"
    if "expert" in model.lower():
        return "expert"
    return "default"


def _extract_vision_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract vision-compatible messages from OpenAI format with auto-compression."""
    import base64, subprocess

    def compress_image(data_url: str, max_size: int = 500 * 1024) -> str:
        """Compress image if base64 data exceeds max_size."""
        if not data_url.startswith("data:image"):
            return data_url
        try:
            header, b64_data = data_url.split(",", 1)
        except ValueError:
            return data_url
        try:
            img_bytes = base64.b64decode(b64_data)
        except Exception:
            return data_url
        if len(img_bytes) <= max_size:
            return data_url

        # Get dimensions without full decode
        w, h = None, None
        if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            import struct
            w = struct.unpack(">I", img_bytes[16:20])[0]
            h = struct.unpack(">I", img_bytes[20:24])[0]
            fmt = 'png'
        elif img_bytes[:3] == b'\xff\xd8\xff':
            import struct
            i = 2
            while i < len(img_bytes) - 1:
                if img_bytes[i] != 0xff: i += 1; continue
                marker = img_bytes[i+1]
                if marker in (0xc0, 0xc2):
                    h = struct.unpack(">H", img_bytes[i+5:i+7])[0]
                    w = struct.unpack(">H", img_bytes[i+7:i+9])[0]
                    fmt = 'jpeg'
                    break
                if marker == 0xd9: break
                length = struct.unpack(">H", img_bytes[i+2:i+4])[0]
                i += 2 + length
            else:
                fmt = 'jpeg'
        if w is None or h is None:
            return data_url

        # Scale down iteratively
        scale = 1.0
        orig_size = len(img_bytes)
        while orig_size * (scale ** 2) > max_size and scale > 0.05:
            scale -= 0.05
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        if new_w >= w or new_h >= h:
            return data_url

        try:
            out_fmt = 'mjpeg' if fmt == 'jpeg' else 'png'
            proc = subprocess.run(
                ['ffmpeg', '-y', '-i', 'pipe:0',
                 '-vf', f'scale={new_w}:{new_h}:force_original_aspect_ratio=decrease',
                 '-f', 'image2pipe', '-'],
                input=img_bytes, capture_output=True, timeout=10
            )
            resized = proc.stdout
            if resized and len(resized) < len(img_bytes):
                new_b64 = base64.b64encode(resized).decode()
                new_mime = 'image/jpeg' if fmt == 'jpeg' else 'image/png'
                return f"data:{new_mime};base64,{new_b64}"
        except Exception:
            pass
        return data_url

    result = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            role = "user"
        content = msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            blocks = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif btype == "image_url":
                    img_url = block.get("image_url", {})
                    url = img_url.get("url", "") if isinstance(img_url, dict) else img_url
                    if url:
                        compressed = compress_image(url)
                        blocks.append({"type": "image_url", "image_url": {"url": compressed}})
            if blocks:
                result.append({"role": role, "content": blocks})
    return result


    if is_vision:
        return "vision"
    if "coder" in model.lower():
        return "coder"
    if "expert" in model.lower():
        return "expert"
    return "default"


def _build_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Build prompt text from messages."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(parts)


def _parse_non_stream_response(resp: Dict[str, Any], std_req: Dict[str, Any]) -> Any:
    """Parse non-stream response into a Turn-like object."""
    from assistantturn.turn import Turn, Usage

    choices = resp.get("choices", [])
    text = ""
    thinking = ""

    if choices:
        message = choices[0].get("message", {})
        text = message.get("content", "")
        thinking = message.get("reasoning_content", "")

    return Turn(
        model=std_req.get("model", "deepseek-chat"),
        prompt="",
        text=text,
        thinking=thinking,
        raw_text=text,
        raw_thinking=thinking,
        usage=Usage(output_tokens=len(text) // 4, total_tokens=len(text) // 4),
    )


def _build_chat_completion_response(
    completion_id: str,
    model: str,
    turn: Any,
    tools_raw: Any,
) -> Dict[str, Any]:
    """Build OpenAI chat completion response."""
    from assistantturn.turn import openai_chat_usage, finish_reason_str

    finish_reason = finish_reason_str(turn) if hasattr(turn, 'stop_reason') else "stop"

    message: Dict[str, Any] = {"role": "assistant", "content": turn.text}

    if turn.thinking:
        message["reasoning_content"] = turn.thinking

    # Handle tool calls
    if turn.tool_calls:
        finish_reason = "tool_calls"
        message["tool_calls"] = [
            {
                "id": tc.id or f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments),
                },
            }
            for i, tc in enumerate(turn.tool_calls)
        ]
        message["content"] = None

    resp = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": openai_chat_usage(turn),
    }

    return resp


def _stream_sse(
    response: Any,
    completion_id: str,
    created: int,
    model: str,
    thinking_enabled: bool,
):
    """Stream SSE events from HTTP response."""
    from assistantturn.turn import Turn, Usage

    buffer = ""
    text_buffer = ""
    thinking_buffer = ""
    tool_buffer = ""
    in_tool = False
    finished = False

    try:
        content = response.content if hasattr(response, 'content') else b""
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            # SSE format: data: {...}
            if line.startswith('data:'):
                data_str = line[5:].strip()
                if data_str == '[DONE]':
                    finished = True
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Parse SSE event
                event_type = event.get("type", "")
                delta = event.get("delta", {}) or {}

                if event_type == "thinking" or (delta.get("type") == "thinking"):
                    # Thinking chunk
                    thinking_content = delta.get("content", "") or delta.get("thinking", "")
                    if thinking_content:
                        thinking_buffer += thinking_content
                        choice = {
                            "delta": {"reasoning_content": thinking_content},
                            "index": 0,
                        }
                        yield _make_sse_chunk(completion_id, created, model, [choice])

                elif event_type == "message" or delta.get("content"):
                    # Text chunk
                    text_content = delta.get("content", "")
                    if text_content:
                        text_buffer += text_content
                        choice = {"delta": {"content": text_content}, "index": 0}
                        yield _make_sse_chunk(completion_id, created, model, [choice])

                elif event_type == "tool_call" or delta.get("tool_call"):
                    # Tool call
                    tool_delta = delta.get("tool_call", delta)
                    tool_buffer += json.dumps(tool_delta)
                    choice = {"delta": {"tool_calls": [tool_delta]}, "index": 0}
                    yield _make_sse_chunk(completion_id, created, model, [choice])

                elif event_type == "finish" or event.get("finish_reason"):
                    # Finish
                    finish_reason = event.get("finish_reason", "stop")
                    choice = {"delta": {}, "index": 0, "finish_reason": finish_reason}
                    yield _make_sse_chunk(completion_id, created, model, [choice])
                    finished = True
                    break

    except Exception as e:
        yield _make_error_chunk(str(e))

    if not finished:
        # Send final finish
        yield _make_sse_chunk(
            completion_id, created, model,
            [{"delta": {}, "index": 0, "finish_reason": "stop"}]
        )


def _make_sse_chunk(
    completion_id: str,
    created: int,
    model: str,
    choices: List[Dict[str, Any]],
) -> str:
    """Make SSE data line."""
    data = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    return f"data: {json.dumps(data)}\n\n"


def _make_error_chunk(message: str) -> str:
    """Make SSE error chunk."""
    data = {"error": {"message": message, "code": "stream_error"}}
    return f"data: {json.dumps(data)}\n\n"


def _make_id() -> str:
    """Generate a completion ID."""
    import uuid
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


# ─── Error Writers ───────────────────────────────────────────────────────────


def write_openai_error(status: int, message: str, code: str = "") -> Dict[str, Any]:
    return {
        "type": "error",
        "status": status,
        "error": {
            "message": message,
            "type": _error_type(status),
            "code": code or _error_code(status),
            "param": None,
            "request_id": _make_id(),
        },
    }


def _error_type(status: int) -> str:
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 429:
        return "rate_limit_error"
    if status >= 500:
        return "server_error"
    return "invalid_request_error"


def _error_code(status: int) -> str:
    codes = {
        400: "invalid_request",
        401: "invalid_api_key",
        403: "permission_denied",
        404: "not_found",
        422: "validation_error",
        429: "rate_limit_exceeded",
        500: "internal_error",
        503: "service_unavailable",
    }
    return codes.get(status, "internal_error")
