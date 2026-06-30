"""
Streaming completion runtime with retry logic

Python port of Go streaming completion runtime.
Handles streaming completions with automatic retry on model_length limit.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, List, Optional, Set, Dict

from config.logger import get_logger

logger = get_logger("completionruntime.stream_retry")


@dataclass
class StreamRetryOptions:
    """Options for streaming retry runtime."""
    surface: str = "api"
    stream: bool = True
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    max_attempts: int = 3
    usage_prompt: str = ""
    request: Any = None
    current_input_file: Any = None


@dataclass
class StreamRetryHooks:
    """Hooks for streaming retry callbacks."""
    consume_attempt: Optional[Callable[[int, Any], None]] = None
    before_retry: Optional[Callable[[int, Any, int], None]] = None
    after_retry: Optional[Callable[[int, Any], None]] = None
    on_error: Optional[Callable[[str], None]] = None


# Sentinel for model length finish reason
FINISH_REASON_MODEL_LENGTH = "model_length"


@dataclass
class StreamChunk:
    """A single chunk from the stream."""
    index: int = 0
    content: str = ""
    thinking: str = ""
    tool_calls: List[Any] = field(default_factory=list)
    finish_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    is_final: bool = False
    raw: Optional[dict] = None


class StreamRetryRuntime:
    """
    Runtime for streaming completions with retry logic.

    Handles streaming responses and automatically retries when
    the model hits its length limit (finish_reason=model_length).
    """

    def __init__(
        self,
        completion_func: Callable[..., Generator],
        continue_func: Callable[..., Any],
        options: StreamRetryOptions,
        hooks: Optional[StreamRetryHooks] = None,
    ):
        self.completion_func = completion_func
        self.continue_func = continue_func
        self.options = options
        self.hooks = hooks or StreamRetryHooks()
        self._retry_count = 0
        self._accumulated_content = ""
        self._accumulated_thinking = ""
        self._accumulated_usage: dict = {}

    def run(self) -> Generator[StreamChunk, None, None]:
        """
        Execute streaming completion with automatic retry.

        Yields:
            StreamChunk events
        """
        session_id = self._get_session_id()
        messages = self._get_messages()
        parent_message_id = self._get_parent_message_id()

        while self._retry_count <= self.options.retry_max_attempts:
            try:
                # Call completion or continue
                if self._retry_count == 0:
                    # First attempt - call completion
                    stream = self.completion_func(
                        session_id=session_id,
                        messages=messages,
                        model=self.options.request.get("model") if self.options.request else "deepseek-chat",
                        stream=True,
                        thinking_enabled=self._get_thinking_enabled(),
                        max_tokens=self._get_max_tokens(),
                        temperature=self.options.request.get("temperature") if self.options.request else 1.0,
                        pow_token=self._get_pow_token(),
                    )
                else:
                    # Retry - call continue
                    if not parent_message_id:
                        logger.warning("No parent_message_id for continue, breaking")
                        break

                    logger.info(f"Continue attempt {self._retry_count}")
                    if self.hooks.before_retry:
                        self.hooks.before_retry(
                            self._retry_count,
                            self.options.request,
                            parent_message_id,
                        )

                    stream = self.continue_func(
                        session_id=session_id,
                        parent_message_id=parent_message_id,
                        stream=True,
                        thinking_enabled=self._get_thinking_enabled(),
                    )

                # Process stream
                for chunk in self._process_stream(stream):
                    yield chunk

                # Stream completed successfully
                break

            except Exception as e:
                logger.error(f"Stream error: {e}")
                if self.hooks.on_error:
                    self.hooks.on_error(str(e))
                break

        if self.hooks.after_retry:
            self.hooks.after_retry(self._retry_count, self.options.request)

    def _process_stream(
        self,
        stream: Generator,
    ) -> Generator[StreamChunk, None, None]:
        """Process streaming events."""
        chunk_index = 0

        for event in stream:
            if isinstance(event, dict):
                if "error" in event:
                    logger.error(f"Stream error: {event['error']}")
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        raw=event,
                    )
                    return

                # Parse event and yield chunks
                chunk = self._parse_event(event, chunk_index)
                if chunk:
                    # Accumulate for retry
                    self._accumulated_content += chunk.content
                    self._accumulated_thinking += chunk.thinking
                    if chunk.usage:
                        self._accumulated_usage.update(chunk.usage)

                    # Track parent message ID for continue
                    if "message_id" in event:
                        self._set_parent_message_id(event["message_id"])

                    # Check for model_length
                    if chunk.finish_reason == FINISH_REASON_MODEL_LENGTH:
                        self._handle_model_length(chunk, chunk_index)
                        return

                    yield chunk
                    chunk_index += 1

    def _parse_event(
        self,
        event: dict,
        index: int,
    ) -> Optional[StreamChunk]:
        """Parse a streaming event into a chunk."""
        choices = event.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason", "")

        content = delta.get("content", "")
        thinking = delta.get("thinking", "") or delta.get("reasoning_content", "")
        tool_calls = delta.get("tool_calls", [])

        usage = event.get("usage", {})

        is_final = finish_reason in ("stop", "length", "model_length")

        return StreamChunk(
            index=index,
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            is_final=is_final,
            raw=event,
        )

    def _handle_model_length(
        self,
        chunk: StreamChunk,
        index: int,
    ) -> None:
        """Handle model_length finish reason - trigger continue."""
        if not self.options.retry_enabled:
            return

        if self._retry_count >= self.options.retry_max_attempts:
            logger.warning("Max retry attempts reached")
            return

        self._retry_count += 1
        logger.info(f"Triggering continue after model_length (attempt {self._retry_count})")

        # Yield the model_length chunk first
        yield_chunk = StreamChunk(
            index=index,
            content="",
            finish_reason="model_length",
            is_final=True,
            raw=chunk.raw,
        )

        # Then continue processing
        self.options.stream = True

        # This will cause the outer run() to call continue_func
        raise RetryException("model_length")

    def _get_session_id(self) -> str:
        """Get session ID from options."""
        if self.options.request:
            return self.options.request.get("session_id", "")
        return ""

    def _get_messages(self) -> list:
        """Get messages from options."""
        if self.options.request:
            return self.options.request.get("messages", [])
        return []

    def _get_parent_message_id(self) -> Optional[int]:
        """Get parent message ID."""
        if self.options.request:
            return self.options.request.get("parent_message_id")
        return None

    def _set_parent_message_id(self, message_id: Any) -> None:
        """Set parent message ID for continue."""
        if self.options.request and message_id:
            self.options.request["parent_message_id"] = message_id

    def _get_thinking_enabled(self) -> bool:
        """Check if thinking is enabled."""
        if self.options.request:
            return self.options.request.get("thinking_enabled", False)
        return False

    def _get_max_tokens(self) -> int:
        """Get max tokens from options."""
        if self.options.request:
            return self.options.request.get("max_tokens", 8192)
        return 8192

    def _get_pow_token(self) -> Optional[str]:
        """Get PoW token from options."""
        if self.options.request:
            return self.options.request.get("pow_token")
        return None


class RetryException(Exception):
    """Signal to retry the stream."""
    pass


def run_stream_with_retry(
    completion_func: Callable,
    continue_func: Callable,
    **kwargs,
) -> Generator[StreamChunk, None, None]:
    """
    Quick helper to run streaming with retry.

    Args:
        completion_func: Function for completion
        continue_func: Function for continue
        **kwargs: Arguments to StreamRetryRuntime

    Yields:
        StreamChunk events
    """
    options = kwargs.pop("options", StreamRetryOptions())
    hooks = kwargs.pop("hooks", None)

    runtime = StreamRetryRuntime(
        completion_func=completion_func,
        continue_func=continue_func,
        options=options,
        hooks=hooks,
        **kwargs,
    )

    retry_count = 0
    max_retries = options.retry_max_attempts

    while retry_count <= max_retries:
        try:
            yield from runtime.run()
            break
        except RetryException:
            retry_count += 1
            if retry_count > max_retries:
                break
