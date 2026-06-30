"""
Assistant Turn - Turn parsing and building from SSE collected results.
"""
import http
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..format.openai import render_chat as openai_render


# ─── Stop Reasons ─────────────────────────────────────────────────────────────


class StopReason:
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


# ─── Usage ────────────────────────────────────────────────────────────────────


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


# ─── Output Error ─────────────────────────────────────────────────────────────


@dataclass
class OutputError:
    status: int = 0
    message: str = ""
    code: str = "error"


# ─── Parsed Tool Call ────────────────────────────────────────────────────────


@dataclass
class ParsedToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""
    raw_json: str = ""


@dataclass
class ToolCallParseResult:
    calls: List[ParsedToolCall] = field(default_factory=list)
    raw_tool_text: str = ""


# ─── Turn ────────────────────────────────────────────────────────────────────


@dataclass
class Turn:
    model: str = ""
    prompt: str = ""
    raw_text: str = ""
    raw_thinking: str = ""
    detection_thinking: str = ""
    text: str = ""
    thinking: str = ""
    tool_calls: List[ParsedToolCall] = field(default_factory=list)
    parsed_tool_calls: ToolCallParseResult = field(default_factory=ToolCallParseResult)
    citation_links: Dict[int, str] = field(default_factory=dict)
    content_filter: bool = False
    response_message_id: int = 0
    stop_reason: str = StopReason.STOP
    usage: Usage = field(default_factory=Usage)
    error: Optional[OutputError] = None


@dataclass
class FinalizeOptions:
    already_emitted_tool_calls: bool = False


@dataclass
class FinalOutcome:
    finish_reason: str = "stop"
    error: Optional[OutputError] = None
    usage: Usage = field(default_factory=Usage)
    has_tool_calls: bool = False
    has_visible_text: bool = False
    has_visible_output: bool = False
    should_fail: bool = False


@dataclass
class BuildOptions:
    model: str = ""
    prompt: str = ""
    ref_file_tokens: int = 0
    search_enabled: bool = False
    strip_reference_markers: bool = False
    tool_names: List[str] = field(default_factory=list)
    tools_raw: Any = None
    tool_choice: Any = None


@dataclass
class StreamSnapshot:
    raw_text: str = ""
    visible_text: str = ""
    raw_thinking: str = ""
    visible_thinking: str = ""
    detection_thinking: str = ""
    content_filter: bool = False
    citation_links: Dict[int, str] = field(default_factory=dict)
    response_message_id: int = 0
    already_emitted_calls: bool = False
    additional_tool_calls: List[ParsedToolCall] = field(default_factory=list)
    already_emitted_tool_raw: bool = False


# ─── SSE Collect Result ──────────────────────────────────────────────────────


class CollectResult:
    """Result collected from SSE stream."""
    def __init__(
        self,
        text: str = "",
        thinking: str = "",
        tool_detection_thinking: str = "",
        content_filter: bool = False,
        citation_links: Optional[Dict[int, str]] = None,
        response_message_id: int = 0,
    ):
        self.text = text
        self.thinking = thinking
        self.tool_detection_thinking = tool_detection_thinking
        self.content_filter = content_filter
        self.citation_links = citation_links or {}
        self.response_message_id = response_message_id


# ─── Build Turn ───────────────────────────────────────────────────────────────


def build_turn_from_collected(result: CollectResult, opts: BuildOptions) -> Turn:
    """Build a Turn from collected SSE data."""
    thinking = _clean_visible_output(result.thinking, opts.strip_reference_markers)
    text = _clean_visible_output(result.text, opts.strip_reference_markers)

    if opts.search_enabled and result.citation_links:
        text = _replace_citation_markers_with_links(text, result.citation_links)

    parsed = _detect_tool_calls(
        result.text, text, result.thinking, result.tool_detection_thinking, opts.tool_names
    )

    calls = _normalize_tool_calls(parsed.calls, opts.tools_raw)
    parsed.calls = calls

    stop_reason = StopReason.STOP
    if result.content_filter:
        stop_reason = StopReason.CONTENT_FILTER
    if calls:
        stop_reason = StopReason.TOOL_CALLS

    turn = Turn(
        model=opts.model,
        prompt=opts.prompt,
        raw_text=result.text,
        raw_thinking=result.thinking,
        detection_thinking=result.tool_detection_thinking,
        text=text,
        thinking=thinking,
        tool_calls=calls,
        parsed_tool_calls=parsed,
        citation_links=result.citation_links,
        content_filter=result.content_filter,
        response_message_id=result.response_message_id,
        stop_reason=stop_reason,
    )
    turn.usage = _build_usage(opts.model, opts.prompt, thinking, text, opts.ref_file_tokens)
    turn.error = _validate_turn(turn, opts.tool_choice)
    if turn.error:
        turn.stop_reason = StopReason.ERROR
    return turn


def build_turn_from_stream_snapshot(snapshot: StreamSnapshot, opts: BuildOptions) -> Turn:
    """Build a Turn from a stream runtime snapshot."""
    thinking = _clean_visible_output(snapshot.visible_thinking, opts.strip_reference_markers)
    text = _clean_visible_output(snapshot.visible_text, opts.strip_reference_markers)

    if opts.search_enabled and snapshot.citation_links:
        text = _replace_citation_markers_with_links(text, snapshot.citation_links)

    parsed = _detect_tool_calls(
        snapshot.raw_text, text, snapshot.raw_thinking,
        snapshot.detection_thinking, opts.tool_names
    )
    calls = parsed.calls
    if not calls and snapshot.additional_tool_calls:
        calls = snapshot.additional_tool_calls
    calls = _normalize_tool_calls(calls, opts.tools_raw)
    parsed.calls = calls

    stop_reason = StopReason.STOP
    if snapshot.content_filter:
        stop_reason = StopReason.CONTENT_FILTER
    if calls or snapshot.already_emitted_calls or snapshot.already_emitted_tool_raw:
        stop_reason = StopReason.TOOL_CALLS

    turn = Turn(
        model=opts.model,
        prompt=opts.prompt,
        raw_text=snapshot.raw_text,
        raw_thinking=snapshot.raw_thinking,
        detection_thinking=snapshot.detection_thinking,
        text=text,
        thinking=thinking,
        tool_calls=calls,
        parsed_tool_calls=parsed,
        citation_links=snapshot.citation_links,
        content_filter=snapshot.content_filter,
        response_message_id=snapshot.response_message_id,
        stop_reason=stop_reason,
    )
    turn.usage = _build_usage(opts.model, opts.prompt, thinking, text, opts.ref_file_tokens)
    if not snapshot.already_emitted_calls and not snapshot.already_emitted_tool_raw:
        turn.error = _validate_turn(turn, opts.tool_choice)
    if turn.error and not calls:
        turn.stop_reason = StopReason.ERROR
    return turn


def _build_usage(model: str, prompt: str, thinking: str, text: str, ref_file_tokens: int) -> Usage:
    """Build usage stats from token counts."""
    input_tokens = _count_tokens(prompt) + ref_file_tokens
    reasoning_tokens = _count_tokens(thinking)
    output_tokens = reasoning_tokens + _count_tokens(text)
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _validate_turn(turn: Turn, tool_choice: Any) -> Optional[OutputError]:
    """Validate turn against tool_choice policy."""
    policy = _resolve_policy(tool_choice)
    if policy.is_required() and not turn.tool_calls:
        return OutputError(
            status=http.HTTPStatus.UNPROCESSABLE_ENTITY,
            message="tool_choice requires at least one valid tool call.",
            code="tool_choice_violation",
        )
    if turn.tool_calls:
        return None
    if turn.text.strip():
        return None
    return _upstream_empty_output_detail(turn.content_filter, turn.text, turn.thinking)


def _upstream_empty_output_detail(content_filter: bool, text: str, thinking: str) -> OutputError:
    """Detail error for upstream empty output."""
    if content_filter:
        return OutputError(
            status=http.HTTPStatus.BAD_REQUEST,
            message="Upstream content filtered the response and returned no output.",
            code="content_filter",
        )
    if thinking.strip():
        return OutputError(
            status=http.HTTPStatus.TOO_MANY_REQUESTS,
            message="Upstream account hit a rate limit and returned reasoning without visible output.",
            code="upstream_empty_output",
        )
    return OutputError(
        status=http.HTTPStatus.SERVICE_UNAVAILABLE,
        message="Upstream service is unavailable and returned no output.",
        code="upstream_unavailable",
    )


def should_retry_empty_output(turn: Turn, attempts: int, max_attempts: int) -> bool:
    """Should retry when turn produced no visible output."""
    return (
        attempts < max_attempts
        and not turn.content_filter
        and not turn.tool_calls
        and not turn.text.strip()
    )


def finalize_turn(turn: Turn, opts: FinalizeOptions) -> FinalOutcome:
    """Finalize turn and determine outcome."""
    has_tool_calls = bool(turn.tool_calls) or opts.already_emitted_tool_calls
    has_visible_text = bool(turn.text.strip())
    has_visible_thinking = bool(turn.thinking.strip())
    err = turn.error

    if has_tool_calls:
        err = None

    finish_reason = finish_reason_str(turn)
    if has_tool_calls:
        finish_reason = "tool_calls"

    return FinalOutcome(
        finish_reason=finish_reason,
        error=err,
        usage=turn.usage,
        has_tool_calls=has_tool_calls,
        has_visible_text=has_visible_text,
        has_visible_output=has_visible_text or has_visible_thinking or has_tool_calls,
        should_fail=err is not None,
    )


def finish_reason_str(turn: Turn) -> str:
    if turn.stop_reason == StopReason.TOOL_CALLS:
        return "tool_calls"
    if turn.stop_reason == StopReason.CONTENT_FILTER:
        return "content_filter"
    return "stop"


def openai_chat_usage(turn: Turn) -> Dict[str, Any]:
    return {
        "prompt_tokens": turn.usage.input_tokens,
        "completion_tokens": turn.usage.output_tokens,
        "total_tokens": turn.usage.total_tokens,
        "completion_tokens_details": {
            "reasoning_tokens": turn.usage.reasoning_tokens,
        },
    }


def openai_responses_usage(turn: Turn) -> Dict[str, Any]:
    return {
        "input_tokens": turn.usage.input_tokens,
        "output_tokens": turn.usage.output_tokens,
        "total_tokens": turn.usage.total_tokens,
    }


# ─── Tool Call Detection ─────────────────────────────────────────────────────


def _detect_tool_calls(
    raw_text: str,
    text: str,
    raw_thinking: str,
    detection_thinking: str,
    tool_names: List[str],
) -> ToolCallParseResult:
    """Detect tool calls from text and thinking."""
    combined = raw_text + raw_thinking + detection_thinking
    calls = _extract_tool_calls_from_text(combined, tool_names)
    return ToolCallParseResult(calls=calls, raw_tool_text=combined)


def _extract_tool_calls_from_text(text: str, tool_names: List[str]) -> List[ParsedToolCall]:
    """Extract tool calls using regex patterns."""
    calls = []

    # Pattern: <tool_calls>...</tool_calls>
    for m in re.finditer(r'<tool_calls>\s*(.*?)\s*</tool_calls>', text, re.DOTALL):
        calls.extend(_parse_tool_block(m.group(1), len(calls)))

    # Pattern: function call syntax
    func_pattern = r'\b(\w+)\s*\(\s*({.*?})\s*\)\s*;?\s*$'
    for m in re.finditer(func_pattern, text, re.MULTILINE | re.DOTALL):
        name = m.group(1)
        args = m.group(2) if m.lastindex >= 2 else "{}"
        if tool_names and name not in tool_names:
            continue
        calls.append(ParsedToolCall(
            id=f"call_{len(calls)}",
            name=name,
            arguments=_ensure_json(args),
            raw_json=m.group(0),
        ))

    return calls[:10]  # Limit to 10 tool calls


def _parse_tool_block(block: str, offset: int) -> List[ParsedToolCall]:
    """Parse a tool_calls block."""
    calls = []
    # Try JSON array
    try:
        arr = json.loads(block)
        if isinstance(arr, list):
            for i, item in enumerate(arr):
                if isinstance(item, dict):
                    calls.append(ParsedToolCall(
                        id=item.get("id", f"call_{offset + i}"),
                        name=item.get("function", {}).get("name", ""),
                        arguments=json.dumps(item.get("function", {}).get("arguments", {})),
                        raw_json=json.dumps(item),
                    ))
            return calls
    except json.JSONDecodeError:
        pass

    # Try extract individual tool calls
    for m in re.finditer(r'"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*({.*?})\s*(?:,|\})', block, re.DOTALL):
        calls.append(ParsedToolCall(
            id=f"call_{offset + len(calls)}",
            name=m.group(1),
            arguments=m.group(2),
            raw_json=m.group(0),
        ))

    return calls


def _normalize_tool_calls(calls: List[ParsedToolCall], tools_raw: Any) -> List[ParsedToolCall]:
    """Normalize tool calls against tool schemas."""
    if not calls:
        return []
    # Deduplicate by argument hash
    seen = set()
    result = []
    for call in calls:
        key = f"{call.name}:{call.arguments[:50]}"
        if key not in seen:
            seen.add(key)
            result.append(call)
    return result


def _resolve_policy(tool_choice: Any) -> 'ToolChoicePolicy':
    """Resolve tool choice policy from raw value."""
    class ToolChoicePolicy:
        def is_none(self): return tool_choice is None or str(tool_choice).lower() == "none"
        def is_required(self): return str(tool_choice).lower() in ("required", "forced")
    return ToolChoicePolicy()


# ─── Text Utilities ──────────────────────────────────────────────────────────


def _clean_visible_output(text: str, strip_markers: bool) -> str:
    """Clean visible output by removing reference markers."""
    if not text:
        return ""
    if strip_markers:
        text = re.sub(r'\[\^[^\^]+\^\]', '', text)
    return text.strip()


def _replace_citation_markers_with_links(text: str, links: Dict[int, str]) -> str:
    """Replace [^[n]] citation markers with actual links."""
    def repl(m):
        idx = int(m.group(1))
        link = links.get(idx, "")
        return f"[{link}]" if link else m.group(0)
    return re.sub(r'\[\^(\d+)\^\]', repl, text)


def _ensure_json(s: str) -> str:
    """Ensure string is valid JSON."""
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        return json.dumps({"raw": s})


def _count_tokens(text: str) -> int:
    """Rough token count (characters / 4)."""
    if not text:
        return 0
    return len(text) // 4
