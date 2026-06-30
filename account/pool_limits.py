"""
Account Pool Limits - Rate limiting utilities
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Dict, Optional

from pool_core import AccountState, Pool


class RateLimitTracker:
    """Track request counts for rate limiting."""

    def __init__(self, window_seconds: int = 60, max_requests: int = 10):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self._lock = Lock()
        self._counts: Dict[str, list] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if a request is allowed for the given key."""
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Clean old entries
            self._counts[key] = [t for t in self._counts[key] if t > cutoff]

            # Check limit
            if len(self._counts[key]) >= self.max_requests:
                return False

            # Record this request
            self._counts[key].append(now)
            return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests for the key."""
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Clean old entries
            self._counts[key] = [t for t in self._counts[key] if t > cutoff]

            return max(0, self.max_requests - len(self._counts[key]))

    def reset(self, key: str) -> None:
        """Reset rate limit for the key."""
        with self._lock:
            if key in self._counts:
                del self._counts[key]


def check_rate_limit(
    account: AccountState,
    window_seconds: int = 60,
    max_requests: int = 10
) -> bool:
    """
    Check if a request is within rate limits for an account.

    Args:
        account: Account to check
        window_seconds: Time window in seconds
        max_requests: Maximum requests per window

    Returns:
        True if allowed, False if rate limited
    """
    # Use account's configured limits if available
    window = getattr(account, "rate_limit_window", None) or window_seconds
    max_req = getattr(account, "rate_limit_requests", None) or max_requests

    # Simple check based on inflight
    if account.inflight >= max_req:
        return False

    return True


def is_excluded(account: AccountState) -> bool:
    """Check if account is currently excluded."""
    return not account.is_available()


def can_use_account(account: AccountState) -> bool:
    """Check if an account can be used for a request."""
    return account.is_available()


def select_best_account(accounts: list) -> Optional[AccountState]:
    """
    Select the best account from a list based on priority and load.

    Args:
        accounts: List of available accounts

    Returns:
        Best AccountState or None
    """
    if not accounts:
        return None

    # Sort by: priority (desc), inflight (asc), last_used (asc)
    accounts.sort(key=lambda a: (-a.priority, a.inflight, a.last_used))
    return accounts[0]
