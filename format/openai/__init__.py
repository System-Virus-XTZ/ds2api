"""OpenAI format rendering."""
from .render_chat import (
    BuildChatCompletion as build_chat_completion,
    BuildChatCompletionWithToolCalls as build_chat_completion_with_tool_calls,
    build_chat_stream_chunk,
    build_chat_stream_delta_choice,
    build_chat_stream_finish_choice,
)
from .render_stream_events import (
    build_responses_created_payload,
    build_responses_output_item_added_payload,
    build_responses_output_item_done_payload,
    build_responses_text_delta_payload,
    build_responses_reasoning_delta_payload,
    build_responses_completed_payload,
    build_responses_failed_payload,
)

__all__ = [
    "build_chat_completion",
    "build_chat_completion_with_tool_calls",
    "build_chat_stream_chunk",
    "build_chat_stream_delta_choice",
    "build_chat_stream_finish_choice",
    "build_responses_created_payload",
    "build_responses_output_item_added_payload",
    "build_responses_output_item_done_payload",
    "build_responses_text_delta_payload",
    "build_responses_reasoning_delta_payload",
    "build_responses_completed_payload",
    "build_responses_failed_payload",
]
