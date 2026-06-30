"""
Account Pool Waiters - Request waiting queue management
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional

from pool_core import AccountState, Pool
from config.logger import get_logger

logger = get_logger("account_pool.waiters")


@dataclass
class Waiter:
    """
    A waiter waiting for an account to become available.
    """
    id: str
    timeout_at: float
    callback: Optional[Callable[[AccountState], None]] = None
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[AccountState] = field(default=None, init=False)


class WaiterManager:
    """
    Manages waiters for available accounts.

    Provides a higher-level interface for waiting on account availability
    with callbacks and automatic cleanup.
    """

    def __init__(self, pool: Pool):
        self._pool = pool
        self._lock = threading.RLock()
        self._waiters: Deque[Waiter] = []
        self._next_id = 0

    def wait(
        self,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[AccountState], None]] = None,
        exclude_ids: Optional[set] = None
    ) -> Optional[AccountState]:
        """
        Wait for an available account.

        Args:
            timeout: Maximum wait time in seconds
            callback: Optional callback when account becomes available
            exclude_ids: Accounts to exclude

        Returns:
            AccountState or None if timeout
        """
        timeout = timeout or self._pool.store.wait_timeout()
        deadline = time.time() + timeout

        waiter = Waiter(
            id=str(self._next_id),
            timeout_at=deadline,
            callback=callback,
        )
        self._next_id += 1

        with self._lock:
            self._waiters.append(waiter)

        # Try to acquire without blocking first
        account = self._pool.acquire(exclude_ids)
        if account:
            waiter.result = account
            waiter.event.set()
            if callback:
                callback(account)
            return account

        # Wait for signal
        remaining = max(0, deadline - time.time())
        signaled = waiter.event.wait(remaining)

        with self._lock:
            # Clean up
            if waiter in self._waiters:
                self._waiters.remove(waiter)

        if signaled and waiter.result:
            if callback:
                callback(waiter.result)
            return waiter.result

        return None

    def notify(self) -> None:
        """
        Notify all waiters that an account might be available.
        Called when an account is released.
        """
        with self._lock:
            self._cleanup_expired()
            self._notify_next()

    def _cleanup_expired(self) -> None:
        """Remove expired waiters."""
        now = time.time()
        expired = [w for w in self._waiters if now > w.timeout_at]
        for waiter in expired:
            waiter.event.set()
            if waiter in self._waiters:
                self._waiters.remove(waiter)
            logger.debug(f"Waiter {waiter.id} timed out")

    def _notify_next(self) -> None:
        """Notify the next waiter to try acquiring."""
        if not self._waiters:
            return

        # Try to find an account for the first waiter
        waiter = self._waiters[0]
        if time.time() <= waiter.timeout_at:
            waiter.event.set()

    def cancel_wait(self, waiter_id: str) -> bool:
        """
        Cancel a specific waiter by ID.

        Returns:
            True if found and cancelled
        """
        with self._lock:
            for waiter in self._waiters:
                if waiter.id == waiter_id:
                    waiter.event.set()
                    if waiter in self._waiters:
                        self._waiters.remove(waiter)
                    return True
        return False

    def cancel_all(self) -> int:
        """
        Cancel all waiters.

        Returns:
            Number of waiters cancelled
        """
        with self._lock:
            count = len(self._waiters)
            for waiter in self._waiters:
                waiter.event.set()
            self._waiters.clear()
            return count

    def count(self) -> int:
        """Get number of waiting requests."""
        with self._lock:
            return len(self._waiters)

    def cleanup(self) -> int:
        """
        Clean up expired waiters.

        Returns:
            Number of waiters cleaned up
        """
        with self._lock:
            before = len(self._waiters)
            self._cleanup_expired()
            return before - len(self._waiters)
