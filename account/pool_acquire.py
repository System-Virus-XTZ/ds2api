"""
Account Pool Acquire - Account acquisition with retry logic
"""

import time
from typing import Optional, Set

from pool_core import AccountState, Pool
from config.logger import get_logger

logger = get_logger("account_pool.acquire")


def acquire_account(
    pool: Pool,
    exclude_ids: Optional[Set[str]] = None,
    tried_accounts: Optional[Set[str]] = None
) -> Optional[AccountState]:
    """
    Try to acquire an account without blocking.

    Args:
        pool: Account pool
        exclude_ids: Additional accounts to exclude
        tried_accounts: Accounts already tried in this request

    Returns:
        Acquired AccountState or None
    """
    exclude_ids = (exclude_ids or set()) | (tried_accounts or set())
    return pool.acquire(exclude_ids)


def acquire_account_wait(
    pool: Pool,
    exclude_ids: Optional[Set[str]] = None,
    tried_accounts: Optional[Set[str]] = None,
    timeout: Optional[float] = None
) -> Optional[AccountState]:
    """
    Wait for an available account.

    Args:
        pool: Account pool
        exclude_ids: Additional accounts to exclude
        tried_accounts: Accounts already tried in this request
        timeout: Maximum wait time in seconds

    Returns:
        Acquired AccountState or None if timeout
    """
    exclude_ids = (exclude_ids or set()) | (tried_accounts or set())
    return pool.acquire_wait(exclude_ids, timeout)


def acquire_with_retry(
    pool: Pool,
    max_attempts: int = 3,
    retry_delay: float = 1.0,
    exclude_ids: Optional[Set[str]] = None
) -> Optional[AccountState]:
    """
    Try to acquire an account with retries.

    Args:
        pool: Account pool
        max_attempts: Maximum number of attempts
        retry_delay: Delay between attempts in seconds
        exclude_ids: Accounts to exclude

    Returns:
        Acquired AccountState or None
    """
    exclude_ids = exclude_ids or set()

    for attempt in range(max_attempts):
        account = pool.acquire(exclude_ids)
        if account:
            return account

        if attempt < max_attempts - 1:
            time.sleep(retry_delay)

    return None


def release_account(pool: Pool, account: AccountState) -> None:
    """
    Release an account back to the pool.

    Args:
        pool: Account pool
        account: Account to release
    """
    pool.release(account)
    logger.debug(f"Released account {account.id}, inflight: {account.inflight}")
