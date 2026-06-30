"""Assistant turn package - handles assistant response processing."""
from .turn import (
    Turn,
    OutputError,
    Usage,
    StopReason,
    build_turn_from_collected as BuildTurnFromCollected,
    build_turn_from_stream_snapshot as BuildTurnFromStreamSnapshot,
    finalize_turn as FinalizeTurn,
    FinalOutcome,
    BuildOptions,
    should_retry_empty_output as ShouldRetryEmptyOutput,
    openai_chat_usage as OpenAIChatUsage,
    openai_responses_usage as OpenAIResponsesUsage,
    finish_reason_str as FinishReason,
)

__all__ = [
    "Turn",
    "OutputError",
    "Usage",
    "StopReason",
    "BuildTurnFromCollected",
    "BuildTurnFromStreamSnapshot",
    "FinalizeTurn",
    "FinalOutcome",
    "BuildOptions",
    "ShouldRetryEmptyOutput",
    "OpenAIChatUsage",
    "OpenAIResponsesUsage",
    "FinishReason",
]
