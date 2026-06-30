"""
Account types - DeepSeek account configuration
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Account:
    """DeepSeek account configuration."""
    id: str
    email: str = ""
    phone: str = ""
    password: str = ""
    token: str = ""
    refresh_token: str = ""
    proxy: str = ""
    priority: int = 0
    enabled: bool = True
    max_inflight: int = 3
    rate_limit_window: int = 60
    rate_limit_requests: int = 10

    def __post_init__(self):
        if self.max_inflight < 1:
            self.max_inflight = 1


@dataclass
class AccountPoolConfig:
    """Account pool configuration."""
    global_max_inflight: int = 10
    account_max_inflight: int = 3
    wait_timeout: int = 300  # seconds
    cleanup_interval: int = 60  # seconds
