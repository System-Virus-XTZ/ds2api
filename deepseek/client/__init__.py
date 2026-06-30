"""DeepSeek client package."""
from .client_core import Client, RequestFailure, FailureKind

__all__ = ["Client", "RequestFailure", "FailureKind"]
