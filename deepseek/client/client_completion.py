"""
Client completion methods - streaming and non-streaming chat completions
"""

import json
from typing import Any, Dict, Generator, Iterator, Optional

import httpx

from .client_core import Client
from .errors import DeepSeekError, ErrRetryable, is_rate_limit
from .http_helpers import read_response_body
from deepseek.protocol.constants import DeepSeekCompletionURL


class CompletionResponse:
    """Response from a completion request."""

    def __init__(
        self,
        content: str = "",
        thinking: str = "",
        finish_reason: str = "",
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.content = content
        self.thinking = thinking
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.error = error

    @property
    def is_error(self) -> bool:
        return self.error is not None


def completion(
    client: Client,
    token: str,
    session_id: str,
    messages: list,
    model: str = "deepseek-chat",
    stream: bool = False,
    thinking_enabled: bool = False,
    search_enabled: bool = False,
    max_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    stop: Optional[list] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
    pow_token: Optional[str] = None,
) -> Any:
    """
    Send a chat completion request.

    Args:
        client: DeepSeek client
        token: Authentication token
        session_id: Chat session ID
        messages: List of message dicts
        model: Model name
        stream: Enable streaming
        thinking_enabled: Enable thinking mode
        search_enabled: Enable search
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        presence_penalty: Presence penalty
        frequency_penalty: Frequency penalty
        stop: Stop sequences
        tools: Tool definitions
        tool_choice: Tool choice policy
        pow_token: PoW token

    Returns:
        CompletionResponse or iterator for streaming
    """
    payload = _build_completion_payload(
        session_id=session_id,
        messages=messages,
        model=model,
        stream=stream,
        thinking_enabled=thinking_enabled,
        search_enabled=search_enabled,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        pow_token=pow_token,
    )

    http_client = client._get_client()

    headers = client._auth_headers(token)
    headers["Content-Type"] = "application/json"
    if pow_token:
        headers["X-PoW-Token"] = pow_token

    if stream:
        return _stream_completion(http_client, DeepSeekCompletionURL, headers, payload, thinking_enabled)
    else:
        return _nonstream_completion(http_client, DeepSeekCompletionURL, headers, payload, thinking_enabled)


def _build_completion_payload(
    session_id: str,
    messages: list,
    model: str,
    stream: bool,
    thinking_enabled: bool,
    search_enabled: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    stop: Optional[list],
    tools: Optional[list],
    tool_choice: Optional[str],
    pow_token: Optional[str],
) -> Dict[str, Any]:
    """Build completion request payload."""
    payload = {
        "chat_session_id": session_id,
        "model": model,
        "stream": stream,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }

    if thinking_enabled:
        payload["thinking_enabled"] = True

    if search_enabled:
        payload["search_enabled"] = True

    if stop:
        payload["stop"] = stop

    if tools:
        payload["tools"] = tools

    if tool_choice:
        payload["tool_choice"] = tool_choice

    if pow_token:
        try:
            pow_data = json.loads(pow_token)
            payload["pow_token"] = pow_data
        except json.JSONDecodeError:
            pass

    return payload


def _nonstream_completion(
    http_client: httpx.Client,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    thinking_enabled: bool,
) -> CompletionResponse:
    """Handle non-streaming completion."""
    try:
        response = http_client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            return CompletionResponse(
                error=f"HTTP {response.status_code}: {response.text}"
            )

        # Read body
        body = read_response_body(response)
        body_text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)

        # Check if response is a JSON error (not SSE)
        if body_text.strip().startswith("{"):
            try:
                error_data = json.loads(body_text)
                if error_data.get("code", 0) != 0:
                    return CompletionResponse(
                        error=error_data.get("msg", "Unknown API error")
                    )
            except json.JSONDecodeError:
                pass

        # Parse SSE response
        data = _parse_sse_response(body)

        if "error" in data:
            return CompletionResponse(error=data["error"])

        content = data.get("content", "")
        thinking = data.get("thinking", "")
        finish_reason = data.get("finish_reason", "stop")
        usage = data.get("usage", {})

        return CompletionResponse(
            content=content,
            thinking=thinking,
            finish_reason=finish_reason,
            usage=usage,
        )

    except httpx.HTTPError as e:
        return CompletionResponse(error=str(e))


def _stream_completion(
    http_client: httpx.Client,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    thinking_enabled: bool,
) -> Generator[Dict[str, Any], None, None]:
    """Handle streaming completion as a generator."""
    try:
        with http_client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                yield {"error": f"HTTP {response.status_code}"}
                return

            for line in response.iter_lines():
                if not line:
                    continue

                # SSE format: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        continue

    except httpx.HTTPError as e:
        yield {"error": str(e)}


def _parse_sse_response(body: bytes) -> Dict[str, Any]:
    """Parse SSE response body into structured data.
    
    Handles both old format (choices[0].delta.content) and 
    new format ({"p":"response/content","o":"APPEND","v":"text"}).
    """

    text_parts = []
    thinking_parts = []
    finish_reason = "stop"
    token_usage = 0

    for line in body.decode("utf-8", errors="replace").split("\n"):
        line = line.strip()
        if not line.startswith("data:") and not line.startswith("data: "):
            continue

        data_str = line[5:].strip() if line.startswith("data:") else line[6:]
        if not data_str or data_str == "[DONE]":
            continue

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # New format (v2): {"p": "response/content", "o": "APPEND", "v": "text"}
        # Also: {"v": "text"} (without p/o, continuation of content)
        if isinstance(data, dict) and "v" in data and isinstance(data.get("v"), str):
            path = data.get("p", "")
            op = data.get("o", "")
            value = data["v"]
            if path == "response/content" or (not path and not op and value):
                text_parts.append(value)
                continue
            if "thinking" in path and value:
                thinking_parts.append(value)
                continue
            if path == "response/accumulated_token_usage":
                token_usage = value
                continue
            if path == "response/status" and value == "FINISHED":
                finish_reason = "stop"
                continue

        # Old format: choices[0].delta.content
        if "choices" in data:
            for choice in data["choices"]:
                delta = choice.get("delta", {})
                if "content" in delta:
                    text_parts.append(delta["content"])
                if "reasoning_content" in delta:
                    thinking_parts.append(delta["reasoning_content"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        if "usage" in data and data["usage"]:
            return {
                "content": "".join(text_parts),
                "thinking": "".join(thinking_parts),
                "finish_reason": finish_reason,
                "usage": data["usage"],
            }

    return {
        "content": "".join(text_parts),
        "thinking": "".join(thinking_parts),
        "finish_reason": finish_reason,
        "usage": {"total_tokens": token_usage},
    }


def call_completion(
    client: Client,
    token: str,
    session_id: str,
    payload: Dict[str, Any],
    pow_token: Optional[str] = None,
    max_attempts: int = 3,
) -> httpx.Response:
    """
    Call completion API and return raw response.

    Args:
        client: DeepSeek client
        token: Authentication token
        session_id: Chat session ID
        payload: Completion payload
        pow_token: PoW token
        max_attempts: Maximum retry attempts

    Returns:
        httpx.Response

    Raises:
        DeepSeekError: If request fails
    """
    http_client = client._get_client()

    headers = client._auth_headers(token)
    headers["Content-Type"] = "application/json"

    if pow_token:
        headers["X-PoW-Token"] = pow_token

    payload["chat_session_id"] = session_id

    for attempt in range(max_attempts):
        try:
            response = http_client.post(
                DeepSeekCompletionURL,
                json=payload,
                headers=headers,
            )

            if response.status_code == 200:
                return response

            if is_rate_limit(response.status_code, 0, 0):
                if attempt < max_attempts - 1:
                    continue
                raise DeepSeekError("Rate limited", status=429)

            if attempt < max_attempts - 1:
                continue

            return response

        except httpx.HTTPError as e:
            if attempt < max_attempts - 1:
                continue
            raise DeepSeekError(f"Request failed: {e}")

    raise DeepSeekError("Max retries exceeded")
