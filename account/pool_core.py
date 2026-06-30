"""
Account Pool Core - Thread-safe account management with rate limiting
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Set

from config.store import ConfigStore
from config.logger import get_logger

logger = get_logger("account_pool")


@dataclass
class AccountState:
    """Runtime state for an account."""
    id: str
    token: str
    proxy: str = ""
    priority: int = 0
    enabled: bool = True
    max_inflight: int = 3
    inflight: int = 0
    last_used: float = 0.0
    consecutive_errors: int = 0
    excluded_until: float = 0.0

    def is_available(self) -> bool:
        """Check if account can accept requests."""
        if not self.enabled:
            return False
        if self.inflight >= self.max_inflight:
            return False
        if time.time() < self.excluded_until:
            return False
        return True

    def can_acquire(self) -> bool:
        """Check if a request can be acquired."""
        return self.is_available()

    def acquire(self) -> bool:
        """Acquire a slot on this account."""
        if not self.can_acquire():
            return False
        self.inflight += 1
        self.last_used = time.time()
        return True

    def release(self) -> None:
        """Release a slot on this account."""
        if self.inflight > 0:
            self.inflight -= 1

    def exclude(self, duration_seconds: float = 60.0) -> None:
        """Temporarily exclude this account."""
        self.excluded_until = time.time() + duration_seconds
        self.consecutive_errors += 1

    def reset_errors(self) -> None:
        """Reset consecutive error count."""
        self.consecutive_errors = 0
        self.excluded_until = 0.0


@dataclass
class WaitingRequest:
    """A request waiting for an available account."""
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[AccountState] = field(default=None)
    timeout_at: float = 0.0


@dataclass
class Pool:
    """
    Thread-safe account pool with rate limiting.

    Manages multiple DeepSeek accounts with:
    - Per-account rate limiting (max concurrent requests)
    - Global rate limiting
    - Priority-based account selection
    - Request queuing with timeout
    - Account exclusion on errors
    """
    store: ConfigStore = field(default_factory=ConfigStore)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _accounts: Dict[str, AccountState] = field(default_factory=dict, repr=False)
    _waiting: Deque[WaitingRequest] = field(default_factory=deque, repr=False)
    _global_inflight: int = 0
    _initialized: bool = False

    def __post_init__(self):
        self._initialize_accounts()

    def _initialize_accounts(self) -> None:
        """Initialize accounts from config."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._accounts.clear()
            for acc_data in self.store.accounts():
                state = AccountState(
                    id=acc_data.get("id", ""),
                    token=acc_data.get("token", ""),
                    proxy=acc_data.get("proxy", ""),
                    priority=acc_data.get("priority", 0),
                    enabled=acc_data.get("enabled", True),
                    max_inflight=acc_data.get("max_inflight", self.store.account_max_inflight()),
                )
                self._accounts[state.id] = state

            self._initialized = True
            logger.info(f"Pool initialized with {len(self._accounts)} accounts")

    def refresh_accounts(self) -> None:
        """Refresh account configuration from store."""
        self._initialized = False
        self._initialize_accounts()

    def get_account(self, account_id: str) -> Optional[AccountState]:
        """Get account state by ID."""
        with self._lock:
            return self._accounts.get(account_id)

    def get_all_accounts(self) -> List[AccountState]:
        """Get all account states."""
        with self._lock:
            return list(self._accounts.values())

    def available_accounts(self, exclude_ids: Optional[Set[str]] = None) -> List[AccountState]:
        """Get available accounts, sorted by priority."""
        exclude_ids = exclude_ids or set()
        with self._lock:
            available = [
                acc for acc in self._accounts.values()
                if acc.can_acquire() and acc.id not in exclude_ids
            ]
            # Sort by priority (higher first) and last used (older first)
            available.sort(key=lambda a: (-a.priority, a.last_used))
            return available

    def global_can_acquire(self) -> bool:
        """Check if global rate limit allows new request."""
        with self._lock:
            return self._global_inflight < self.store.global_max_inflight()

    def acquire_global(self) -> bool:
        """Acquire global slot."""
        with self._lock:
            if self._global_inflight >= self.store.global_max_inflight():
                return False
            self._global_inflight += 1
            return True

    def release_global(self) -> None:
        """Release global slot."""
        with self._lock:
            if self._global_inflight > 0:
                self._global_inflight -= 1

    def acquire(self, exclude_ids: Optional[Set[str]] = None) -> Optional[AccountState]:
        """
        Try to acquire an account without blocking.

        Args:
            exclude_ids: Account IDs to exclude (e.g., already tried)

        Returns:
            Acquired AccountState or None
        """
        self._initialize_accounts()
        exclude_ids = exclude_ids or set()

        if not self.acquire_global():
            return None

        available = self.available_accounts(exclude_ids)
        if not available:
            self.release_global()
            return None

        account = available[0]
        if account.acquire():
            return account

        self.release_global()
        return None

    def release(self, account: AccountState) -> None:
        """Release an account slot."""
        account.release()
        self.release_global()
        self._notify_waiting()

    def _notify_waiting(self) -> None:
        """Notify waiting requests that an account might be available."""
        while self._waiting:
            wait_req = self._waiting[0]
            # Check if timed out
            if time.time() > wait_req.timeout_at:
                wait_req.event.set()
                self._waiting.popleft()
                continue

            # Try to acquire
            account = self.acquire()
            if account:
                wait_req.result = account
                wait_req.event.set()
                self._waiting.popleft()
                return

            # No available account, stop
            break

    def acquire_wait(
        self,
        exclude_ids: Optional[Set[str]] = None,
        timeout: Optional[float] = None
    ) -> Optional[AccountState]:
        """
        Wait for an available account.

        Args:
            exclude_ids: Account IDs to exclude
            timeout: Maximum wait time in seconds

        Returns:
            Acquired AccountState or None if timeout
        """
        self._initialize_accounts()
        exclude_ids = exclude_ids or set()
        timeout = timeout or self.store.wait_timeout()

        deadline = time.time() + timeout

        # Try immediate acquisition first
        account = self.acquire(exclude_ids)
        if account:
            return account

        # Wait for availability
        wait_req = WaitingRequest(timeout_at=deadline)

        with self._lock:
            self._waiting.append(wait_req)

        # Wait for signal
        signaled = wait_req.event.wait(timeout=max(0, deadline - time.time()))

        if signaled and wait_req.result:
            return wait_req.result

        # Timeout or failed
        with self._lock:
            # Remove from waiting list if still there
            if wait_req in self._waiting:
                self._waiting.remove(wait_req)

        return None

    def wait_for_available(
        self,
        exclude_ids: Optional[Set[str]] = None,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Wait until an account is available.

        This is non-acquiring - just blocks until something is free.
        """
        timeout = timeout or self.store.wait_timeout()
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self.available_accounts(exclude_ids):
                return True
            time.sleep(0.1)

        return False

    def remove(self, account_id: str) -> bool:
        """Remove an account from the pool."""
        with self._lock:
            if account_id in self._accounts:
                del self._accounts[account_id]
                logger.info(f"Removed account {account_id} from pool")
                return True
            return False

    def add(self, account_data: dict) -> bool:
        """Add an account to the pool."""
        state = AccountState(
            id=account_data.get("id", ""),
            token=account_data.get("token", ""),
            proxy=account_data.get("proxy", ""),
            priority=account_data.get("priority", 0),
            enabled=account_data.get("enabled", True),
            max_inflight=account_data.get("max_inflight", self.store.account_max_inflight()),
        )
        with self._lock:
            self._accounts[state.id] = state
            logger.info(f"Added account {state.id} to pool")
            return True

    def stats(self) -> dict:
        """Get pool statistics."""
        with self._lock:
            total = len(self._accounts)
            available = sum(1 for a in self._accounts.values() if a.is_available())
            waiting = len(self._waiting)

            return {
                "total_accounts": total,
                "available_accounts": available,
                "global_inflight": self._global_inflight,
                "global_max": self.store.global_max_inflight(),
                "waiting_requests": waiting,
            }

    def reset_account(self, account_id: str) -> bool:
        """Reset an account's error state."""
        with self._lock:
            if account_id in self._accounts:
                self._accounts[account_id].reset_errors()
                return True
            return False

    def exclude_account(self, account_id: str, duration_seconds: float = 60.0) -> bool:
        """Temporarily exclude an account."""
        with self._lock:
            if account_id in self._accounts:
                self._accounts[account_id].exclude(duration_seconds)
                logger.warning(f"Excluded account {account_id} for {duration_seconds}s")
                return True
            return False
