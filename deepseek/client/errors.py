"""
DeepSeek client errors
"""

import httpx


class DeepSeekError(Exception):
    """Base exception for DeepSeek client errors."""
    def __init__(self, message: str, status: int = 0, code: str = ""):
        self.message = message
        self.status = status
        self.code = code
        super().__init__(message)


class ErrRetryable(DeepSeekError):
    """Error that can be retried."""
    pass


class ErrUnauthorized(DeepSeekError):
    """Authentication/authorization error."""
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message, status=401, code="unauthorized")


class ErrRateLimit(DeepSeekError):
    """Rate limit error."""
    def __init__(self, message: str = "Rate limited"):
        super().__init__(message, status=429, code="rate_limit_exceeded")


class ErrNotFound(DeepSeekError):
    """Resource not found."""
    def __init__(self, message: str = "Not found"):
        super().__init__(message, status=404, code="not_found")


class ErrServerError(DeepSeekError):
    """Server-side error."""
    def __init__(self, message: str = "Server error", status: int = 500):
        super().__init__(message, status=status, code="server_error")


def is_token_invalid(status: int, code: int, biz_code: int, msg: str, biz_msg: str) -> bool:
    """Check if error indicates an invalid token."""
    if status == 401:
        return True
    if code == 401 or biz_code == 401:
        return True
    msg_lower = msg.lower()
    biz_msg_lower = biz_msg.lower()
    if "token" in msg_lower and ("invalid" in msg_lower or "expired" in msg_lower):
        return True
    if "token" in biz_msg_lower and ("invalid" in biz_msg_lower or "expired" in biz_msg_lower):
        return True
    return False


def is_rate_limit(status: int, code: int, biz_code: int) -> bool:
    """Check if error indicates rate limiting."""
    if status == 429:
        return True
    if code == 429 or biz_code == 429:
        return True
    return False


def get_error_message(response: dict) -> str:
    """Extract error message from API response."""
    if msg := response.get("msg"):
        return str(msg)
    if error := response.get("error"):
        return str(error)
    if message := response.get("message"):
        return str(message)
    return "Unknown error"


def wrap_http_error(exc: httpx.HTTPError) -> DeepSeekError:
    """Convert httpx error to DeepSeekError."""
    if isinstance(exc, httpx.TimeoutException):
        return ErrRetryable(f"Request timeout: {exc}", status=504, code="timeout")
    if isinstance(exc, httpx.ConnectError):
        return ErrRetryable(f"Connection error: {exc}", status=503, code="connection_error")
    if isinstance(exc, httpx.HTTPStatusError):
        return DeepSeekError(
            f"HTTP {exc.response.status_code}: {exc}",
            status=exc.response.status_code,
            code="http_error"
        )
    return ErrRetryable(f"HTTP error: {exc}", code="http_error")
