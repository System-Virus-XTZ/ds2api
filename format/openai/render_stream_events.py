"""OpenAI Responses API format stream events."""
import time
import json
from typing import Any, Dict, Optional


def build_responses_created_payload(response_id: str, model: str) -> Dict[str, Any]:
    return {
        "type": "response.created",
        "id": response_id,
        "response_id": response_id,
        "object": "response",
        "model": model,
        "status": "in_progress",
    }


def build_responses_output_item_added_payload(
    response_id: str, item_id: str, output_index: int, item: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "id": response_id,
        "response_id": response_id,
        "output_index": output_index,
        "item_id": item_id,
        "item": item,
    }


def build_responses_output_item_done_payload(
    response_id: str, item_id: str, output_index: int, item: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "id": response_id,
        "response_id": response_id,
        "output_index": output_index,
        "item_id": item_id,
        "item": item,
    }


def build_responses_content_part_added_payload(
    response_id: str, item_id: str, output_index: int, content_index: int, part: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "type": "response.content_part.added",
        "id": response_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": part,
    }


def build_responses_content_part_done_payload(
    response_id: str, item_id: str, output_index: int, content_index: int, part: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "type": "response.content_part.done",
        "id": response_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": part,
    }


def build_responses_text_delta_payload(
    response_id: str, item_id: str, output_index: int, content_index: int, delta: str
) -> Dict[str, Any]:
    return {
        "type": "response.output_text.delta",
        "id": response_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "delta": delta,
    }


def build_responses_text_done_payload(
    response_id: str, item_id: str, output_index: int, content_index: int, text: str
) -> Dict[str, Any]:
    return {
        "type": "response.output_text.done",
        "id": response_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "text": text,
    }


def build_responses_reasoning_delta_payload(response_id: str, delta: str) -> Dict[str, Any]:
    return {
        "type": "response.reasoning.delta",
        "id": response_id,
        "response_id": response_id,
        "delta": delta,
    }


def build_responses_function_call_arguments_delta_payload(
    response_id: str, item_id: str, output_index: int, call_id: str, delta: str
) -> Dict[str, Any]:
    return {
        "type": "response.function_call_arguments.delta",
        "id": response_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "call_id": call_id,
        "delta": delta,
    }


def build_responses_function_call_arguments_done_payload(
    response_id: str, item_id: str, output_index: int, call_id: str, name: str, arguments: str
) -> Dict[str, Any]:
    return {
        "type": "response.function_call_arguments.done",
        "id": response_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "call_id": call_id,
        "name": name,
        "arguments": _normalize_json_string(arguments),
    }


def build_responses_completed_payload(response: Dict[str, Any]) -> Dict[str, Any]:
    response_id = response.get("id", "")
    return {
        "type": "response.completed",
        "response_id": response_id,
        "response": response,
    }


def build_responses_failed_payload(
    response_id: str, model: str, status: int, message: str, code: str = ""
) -> Dict[str, Any]:
    code = code.strip() if code else ""
    if not code:
        code = "api_error"
    return {
        "type": "response.failed",
        "id": response_id,
        "response_id": response_id,
        "object": "response",
        "model": model,
        "status": "failed",
        "status_code": status,
        "error": {
            "message": message,
            "type": _responses_error_type(status),
            "code": code,
            "param": None,
        },
    }


def _responses_error_type(status: int) -> str:
    if status in {400, 404, 422}:
        return "invalid_request_error"
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 429:
        return "rate_limit_error"
    if status == 503:
        return "service_unavailable_error"
    if status >= 500:
        return "api_error"
    return "invalid_request_error"


def _normalize_json_string(s: str) -> str:
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        return s.strip()
