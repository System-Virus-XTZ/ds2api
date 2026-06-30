"""
OpenAI format chat rendering

Python port of Go OpenAI format rendering for chat completions.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from config.logger import get_logger

logger = get_logger("format.openai.render_chat")


def generate_id() -> str:
    """Generate a unique ID for completion/response."""
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return f"msg_{uuid.uuid4().hex[:24]}"


def BuildChatCompletion(
    completion_id: str,
    model: str,
    final_prompt: str,
    final_thinking: str,
    final_text: str,
    tool_names: List[str],
    tools_raw: Any,
) -> Dict[str, Any]:
    """
    Build a chat completion response.

    Args:
        completion_id: Completion ID
        model: Model name
        final_prompt: The final prompt used
        final_thinking: Reasoning/thinking content
        final_text: The final text response
        tool_names: List of available tool names
        tools_raw: Raw tools definition

    Returns:
        OpenAI format completion response dict
    """
    # Parse tool calls from response
    detected_calls = _parse_tool_calls(final_text, final_thinking, tool_names)

    return BuildChatCompletionWithToolCalls(
        completion_id=completion_id,
        model=model,
        final_prompt=final_prompt,
        final_thinking=final_thinking,
        final_text=final_text,
        detected=detected_calls,
        tools_raw=tools_raw,
    )


def BuildChatCompletionWithToolCalls(
    completion_id: str,
    model: str,
    final_prompt: str,
    final_thinking: str,
    final_text: str,
    detected: List[Dict[str, Any]],
    tools_raw: Any,
) -> Dict[str, Any]:
    """
    Build a chat completion with parsed tool calls.

    Args:
        completion_id: Completion ID
        model: Model name
        final_prompt: The final prompt
        final_thinking: Reasoning content
        final_text: The final text
        detected: Parsed tool calls
        tools_raw: Raw tools definition

    Returns:
        OpenAI format response dict
    """
    now = int(time.time())
    message_id = generate_message_id()

    # Build message
    message = {
        "role": "assistant",
        "content": final_text,
        "tool_calls": [],
    }

    if detected:
        for i, call in enumerate(detected):
            message["tool_calls"].append({
                "index": i,
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": call.get("arguments", "{}"),
                },
            })

    # Build response
    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": now,
        "model": model,
        "system_fingerprint": "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": "tool_calls" if detected else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    return response


def _parse_tool_calls(
    text: str,
    thinking: str,
    tool_names: List[str],
) -> List[Dict[str, Any]]:
    """Parse tool calls from text and thinking content."""
    import json
    import re

    detected = []

    # Look for tool call patterns in text
    patterns = [
        r'<tool_call>\s*\{([^}]+)\}',
        r'"name"\s*:\s*"(\w+)"[^}]*"arguments"\s*:\s*(\{[^}]+\}|\"[^\"]+\")',
        r'```json\s*\n([\s\S]*?)\n```',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                if "{" in match.group(0):
                    # Try to parse as JSON
                    json_str = match.group(0)
                    if json_str.startswith("```json"):
                        json_str = re.sub(r'```json\s*\n?', '', json_str)
                        json_str = re.sub(r'\n?\s*```', '', json_str)

                    data = json.loads(json_str)
                    if "name" in data:
                        detected.append({
                            "name": data["name"],
                            "arguments": data.get("arguments", "{}"),
                        })
            except (json.JSONDecodeError, IndexError):
                continue

    # Also check thinking content
    for pattern in patterns:
        matches = re.finditer(pattern, thinking, re.DOTALL)
        for match in matches:
            try:
                if "{" in match.group(0):
                    json_str = match.group(0)
                    if json_str.startswith("```json"):
                        json_str = re.sub(r'```json\s*\n?', '', json_str)
                        json_str = re.sub(r'\n?\s*```', '', json_str)

                    data = json.loads(json_str)
                    if "name" in data and data["name"] not in [d["name"] for d in detected]:
                        detected.append({
                            "name": data["name"],
                            "arguments": data.get("arguments", "{}"),
                        })
            except (json.JSONDecodeError, IndexError):
                continue

    return detected


def format_usage(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> Dict[str, int]:
    """Format usage statistics."""
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": 0,
        } if reasoning_tokens else None,
    }


# ─── Streaming helpers (exported with snake_case) ────────────────────────────


def build_chat_stream_chunk(
    completion_id: str,
    created: int,
    model: str,
    choices: list,
    usage: dict = None,
) -> dict:
    """Build a streaming chunk event data (OpenAI format)."""
    out = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    if usage:
        out["usage"] = usage
    return out


def build_chat_stream_delta_choice(index: int, delta: dict) -> dict:
    """Build a streaming delta choice."""
    return {"delta": delta, "index": index}


def build_chat_stream_finish_choice(index: int, finish_reason: str) -> dict:
    """Build a streaming finish choice."""
    return {"delta": {}, "index": index, "finish_reason": finish_reason}
