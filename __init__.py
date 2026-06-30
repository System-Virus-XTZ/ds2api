"""
DS2API

DeepSeek API Proxy with multi-account pooling, chat history, and OpenAI-compatible API.
"""

__version__ = "1.0.0"
__author__ = "DS2API Team"

from config.store import ConfigStore
from auth.request import RequestAuth, AuthResolver
from account.pool_core import Pool, AccountState
from deepseek.client.client_core import Client
from chathistory.store import ChatHistoryStore

__all__ = [
    "__version__",
    "ConfigStore",
    "RequestAuth",
    "AuthResolver",
    "Pool",
    "AccountState",
    "Client",
    "ChatHistoryStore",
]
