"""
SSE (Server-Sent Events) utilities

Python port of Go SSE parsing utilities for DeepSeek streaming responses.
"""

import json
from typing import Callable, Generator, Iterator, Optional


def parse_sse_line(line: bytes) -> Optional[str]:
    """
    Parse a single SSE line.

    Args:
        line: Raw line bytes

    Returns:
        Data string without "data: " prefix, or None if not a data line
    """
    line = line.rstrip(b"\r\n")

    if not line:
        return None

    if line.startswith(b"data: "):
        return line[6:].decode("utf-8", errors="replace")

    return None


def parse_sse_data(data: str) -> Optional[dict]:
    """
    Parse SSE data as JSON.

    Args:
        data: Data string (may be "[DONE]")

    Returns:
        Parsed dict or None if not JSON
    """
    if data == "[DONE]":
        return {"done": True}

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def scan_sse_lines(
    response,
    on_line: Callable[[bytes], bool],
) -> None:
    """
    Scan SSE lines from a response.

    Args:
        response: HTTP response object with iter_lines()
        on_line: Callback for each line. Return False to stop.

    Yields:
        Data strings from "data: " lines
    """
    for line in response.iter_lines():
        if not line:
            continue

        data = parse_sse_line(line)
        if data is not None:
            if not on_line(line):
                return


def iter_sse_events(
    response,
) -> Generator[dict, None, None]:
    """
    Iterate SSE events from a response.

    Args:
        response: HTTP response object

    Yields:
        Parsed event dicts
    """
    for line in response.iter_lines():
        if not line:
            continue

        data = parse_sse_line(line)
        if data is None:
            continue

        event = parse_sse_data(data)
        if event is not None:
            yield event


class SSEParser:
    """SSE event parser with state tracking."""

    def __init__(self):
        self._event_type = ""
        self._data = []

    def feed_line(self, line: bytes) -> Optional[dict]:
        """
        Feed a single line to the parser.

        Returns:
            Complete event dict when an event is finished, or None
        """
        line = line.rstrip(b"\r\n")

        if not line:
            # Empty line = event end
            if self._data:
                event = self._build_event()
                self._reset()
                return event
            return None

        if line.startswith(b":"):
            # Comment line
            return None

        # Parse field
        if b":" in line:
            field, value = line.split(b":", 1)
            field = field.strip().decode("utf-8", errors="replace")
            value = value.strip()
            if value.startswith(b" "):
                value = value[1:]
            value = value.decode("utf-8", errors="replace")

            if field == "event":
                self._event_type = value
            elif field == "data":
                self._data.append(value)

        return None

    def _build_event(self) -> dict:
        """Build event dict from collected data."""
        return {
            "event": self._event_type,
            "data": "\n".join(self._data),
        }

    def _reset(self) -> None:
        """Reset parser state."""
        self._event_type = ""
        self._data = []


def is_done_event(data: dict) -> bool:
    """Check if this is a [DONE] event."""
    return data.get("done") or data.get("data") == "[DONE]"
