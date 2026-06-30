"""
Client continue completion - continue a message with model length limit
"""

import json
from typing import Any, Dict, Generator, Optional

import httpx

from .client_core import Client
from .errors import DeepSeekError
from .http_helpers import read_response_body
from deepseek.protocol.constants import DeepSeekContinueURL


def continue_completion(
    client: Client,
    token: str,
    session_id: str,
    parent_message_id: int,
    stream: bool = False,
    thinking_enabled: bool = False,
) -> Any:
    """
    Continue a message that was truncated due to model length limit.

    Args:
        client: DeepSeek client
        token: Authentication token
        session_id: Chat session ID
        parent_message_id: ID of the message to continue
        stream: Enable streaming
        thinking_enabled: Enable thinking mode

    Returns:
        Completion response or iterator for streaming
    """
    payload = {
        "chat_session_id": session_id,
        "parent_message_id": parent_message_id,
        "stream": stream,
    }

    if thinking_enabled:
        payload["thinking_enabled"] = True

    http_client = client._get_client()
    headers = client._auth_headers(token)
    headers["Content-Type"] = "application/json"

    if stream:
        return _stream_continue(http_client, headers, payload)
    else:
        return _nonstream_continue(http_client, headers, payload)


def _nonstream_continue(
    http_client: httpx.Client,
    headers: Dict[str, str],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle non-streaming continue."""
    try:
        response = http_client.post(
            DeepSeekContinueURL,
            json=payload,
            headers=headers,
        )

        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text}"}

        body = read_response_body(response)
        data = json.loads(body.decode("utf-8", errors="replace"))

        return data

    except httpx.HTTPError as e:
        return {"error": str(e)}


def _stream_continue(
    http_client: httpx.Client,
    headers: Dict[str, str],
    payload: Dict[str, Any],
) -> Generator[Dict[str, Any], None, None]:
    """Handle streaming continue as a generator."""
    try:
        with http_client.stream(
            "POST",
            DeepSeekContinueURL,
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                yield {"error": f"HTTP {response.status_code}"}
                return

            for line in response.iter_lines():
                if not line:
                    continue

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


def call_continue(
    client: Client,
    token: str,
    session_id: str,
    parent_message_id: int,
    thinking_enabled: bool = False,
) -> httpx.Response:
    """
    Call continue API and return raw response.

    Args:
        client: DeepSeek client
        token: Authentication token
        session_id: Chat session ID
        parent_message_id: ID of the message to continue
        thinking_enabled: Enable thinking mode

    Returns:
        httpx.Response
    """
    http_client = client._get_client()
    headers = client._auth_headers(token)
    headers["Content-Type"] = "application/json"

    payload = {
        "chat_session_id": session_id,
        "parent_message_id": parent_message_id,
        "stream": False,
    }

    if thinking_enabled:
        payload["thinking_enabled"] = True

    response = http_client.post(
        DeepSeekContinueURL,
        json=payload,
        headers=headers,
    )

    return response
