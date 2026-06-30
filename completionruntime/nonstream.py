"""
Non-streaming completion runtime

Python port of Go non-streaming completion runtime.
Handles single-shot completion requests that wait for the full response.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from config.logger import get_logger

logger = get_logger("completionruntime.nonstream")


@dataclass
class NonStreamResult:
    """Result of non-streaming completion."""
    content: str = ""
    thinking: str = ""
    finish_reason: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: int = 0


class NonStreamRuntime:
    """
    Runtime for non-streaming completions.

    Handles single-shot completion requests that wait for the full
    response before returning.
    """

    def __init__(
        self,
        completion_func: Callable[..., Any],
        session_id: str,
        messages: list,
        model: str,
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
    ):
        self.completion_func = completion_func
        self.session_id = session_id
        self.messages = messages
        self.model = model
        self.thinking_enabled = thinking_enabled
        self.search_enabled = search_enabled
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.tools = tools
        self.tool_choice = tool_choice
        self.pow_token = pow_token

    def run(self) -> NonStreamResult:
        """
        Execute the completion and return result.

        Returns:
            NonStreamResult with content and metadata
        """
        start_time = time.time()

        try:
            response = self.completion_func(
                session_id=self.session_id,
                messages=self.messages,
                model=self.model,
                stream=False,
                thinking_enabled=self.thinking_enabled,
                search_enabled=self.search_enabled,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                stop=self.stop,
                tools=self.tools,
                tool_choice=self.tool_choice,
                pow_token=self.pow_token,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            if hasattr(response, "is_error") and response.is_error:
                return NonStreamResult(
                    error=response.error,
                    duration_ms=duration_ms,
                )

            return NonStreamResult(
                content=response.content if hasattr(response, "content") else str(response),
                thinking=response.thinking if hasattr(response, "thinking") else "",
                finish_reason=response.finish_reason if hasattr(response, "finish_reason") else "stop",
                usage=response.usage if hasattr(response, "usage") else {},
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"Non-stream completion failed: {e}")
            duration_ms = int((time.time() - start_time) * 1000)
            return NonStreamResult(
                error=str(e),
                duration_ms=duration_ms,
            )


def run_nonstream(
    completion_func: Callable,
    **kwargs,
) -> NonStreamResult:
    """
    Quick helper to run a non-streaming completion.

    Args:
        completion_func: Function to call for completion
        **kwargs: Arguments to pass to completion

    Returns:
        NonStreamResult
    """
    runtime = NonStreamRuntime(completion_func=completion_func, **kwargs)
    return runtime.run()
