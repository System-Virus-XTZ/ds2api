"""
Request authentication - API key and account resolution
"""

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from config.store import ConfigStore, Store


@dataclass
class RequestAuth:
    """
    Request authentication context.

    Contains the resolved DeepSeek token and account information
    for the current request.
    """
    DeepSeekToken: str = ""
    AccountID: str = ""
    UseConfigToken: bool = False
    CallerID: str = ""
    SourceHeader: str = ""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _ref_count: int = field(default=0, repr=False)

    @property
    def deepseek_token(self) -> str:
        """Alias for DeepSeekToken (snake_case compatibility)."""
        return self.DeepSeekToken

    @deepseek_token.setter
    def deepseek_token(self, value: str) -> None:
        self.DeepSeekToken = value

    @property
    def account_id(self) -> str:
        """Alias for AccountID (snake_case compatibility)."""
        return self.AccountID

    @account_id.setter
    def account_id(self, value: str) -> None:
        self.AccountID = value

    def add_ref(self) -> None:
        """Increment reference count."""
        with self._lock:
            self._ref_count += 1

    def release(self) -> int:
        """Decrement reference count and return remaining count."""
        with self._lock:
            self._ref_count -= 1
            return self._ref_count


@dataclass
class AuthResolver:
    """
    Resolves authentication for incoming requests.

    Supports:
    - API key authentication (X-API-Key header)
    - Bearer token authentication (Authorization header)
    - Account targeting (X-Ds2-Target-Account header)
    """
    store: Store
    pool: "Pool" = field(default=None, repr=False)
    login_func: Optional[Callable] = field(default=None, repr=False)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def determine(self, request, skip_deepseek: bool = False) -> RequestAuth:
        """
        Determine authentication from request.

        Args:
            request: HTTP request object (Starlette Request or dict)

        Returns:
            RequestAuth with resolved credentials

        Raises:
            ValueError: If authentication fails
        """
        headers = self._extract_headers(request)

        # Get API key from various sources
        api_key = (
            headers.get("x-api-key") or
            headers.get("authorization", "").replace("Bearer ", "") or
            ""
        )

        caller_id = headers.get("x-ds2-source", "")
        target_account = headers.get("x-ds2-target-account", "")

        # No API key: auto-acquire account from pool and login
        if not api_key:
            return self._auto_auth_from_pool(caller_id)

        # Check if it's an external API key (sk-...)
        if api_key and self._is_external_api_key(api_key):
            return self._auto_auth_from_pool(caller_id)

        # Check if it's a DS2API internal key
        if self._is_ds2api_key(api_key):
            return self._resolve_ds2api_key(api_key, caller_id, target_account)

        # Try to find account by key
        if self.store.key_exists(api_key):
            return self._resolve_account_key(api_key, caller_id, target_account)

        # Fall back to account targeting
        if target_account:
            return self._resolve_target_account(target_account, caller_id)

        raise ValueError("No valid authentication provided")

    def determine_caller(self, request) -> RequestAuth:
        """Determine caller identity without account allocation."""
        auth = self.determine(request)
        return auth

    def _is_ds2api_key(self, key: str) -> bool:
        """Check if this is a DS2API internal key."""
        return key.startswith("ds2api-") or key.startswith("sk-ds2-")

    def _is_external_api_key(self, key: str) -> bool:
        """Check if this is an external API key (sk-...)."""
        return key.startswith("sk-") and not self._is_ds2api_key(key)

    def _resolve_ds2api_key(
        self, key: str, caller_id: str, target_account: str
    ) -> RequestAuth:
        """Resolve authentication for a DS2API key."""
        # DS2API keys map to configured accounts
        # For now, use the first available account or target account
        if target_account:
            account = self.store.account_by_id(target_account)
            if account:
                return self._account_to_auth(account, caller_id)

        # Find first enabled account
        for account in self.store.accounts():
            if account.get("enabled", True):
                return self._account_to_auth(account, caller_id)

        raise ValueError("No available accounts")

    def _resolve_account_key(
        self, key: str, caller_id: str, target_account: str
    ) -> RequestAuth:
        """Resolve authentication for an account-mapped API key."""
        # In the current design, keys are global, not per-account
        if target_account:
            account = self.store.account_by_id(target_account)
            if account:
                return self._account_to_auth(account, caller_id)

        # Use first available account
        for account in self.store.accounts():
            if account.get("enabled", True):
                return self._account_to_auth(account, caller_id)

        raise ValueError("No available accounts")

    def _resolve_target_account(
        self, account_id: str, caller_id: str
    ) -> RequestAuth:
        """Resolve authentication for targeted account."""
        account = self.store.account_by_id(account_id)
        if not account:
            raise ValueError(f"Account not found: {account_id}")

        if not account.get("enabled", True):
            raise ValueError(f"Account disabled: {account_id}")

        return self._account_to_auth(account, caller_id)

    def _auto_auth_from_pool(self, caller_id: str) -> RequestAuth:
        """Auto-acquire an account from the pool and login automatically."""
        if not self.pool:
            raise ValueError("No account pool available")

        # Acquire an available account from pool
        account_state = self.pool.acquire()
        if not account_state:
            raise ValueError("No available accounts in pool")

        try:
            account = self.store.account_by_id(account_state.id)
            if not account:
                raise ValueError(f"Account not found: {account_state.id}")

            token = ""
            if self.login_func:
                try:
                    token = self.login_func(account)
                except Exception as e:
                    import logging
                    logging.getLogger("ds2api.auth").warning(
                        f"Auto-login failed for {account.get('email')}: {e}"
                    )
                    raise ValueError(f"Login failed: {e}")

            if not token:
                raise ValueError("Login returned empty token")

            return RequestAuth(
                DeepSeekToken=token,
                AccountID=account.get("id", ""),
                UseConfigToken=False,
                CallerID=caller_id,
                SourceHeader="auto-pool",
            )
        finally:
            # Release account back to pool after resolving token
            # (pool will handle concurrent use via its own inflight tracking)
            self.pool.release(account_state)

    def _extract_headers(self, request) -> dict:
        """Extract headers from request object."""
        if isinstance(request, dict):
            return {k.lower(): v for k, v in request.get("headers", {}).items()}

        # Assume it's a Starlette Request
        if hasattr(request, "headers"):
            return {k.lower(): v for k, v in request.headers.items()}

        return {}

    def release(self, auth: RequestAuth) -> None:
        """Release authentication context."""
        # In this implementation, release is a no-op
        # The account pool handles release separately
        pass

    def switch_account(self, auth: RequestAuth) -> bool:
        """Switch to a different account."""
        # Find next available account
        for account in self.store.accounts():
            if account.get("id") != auth.AccountID and account.get("enabled", True):
                new_auth = self._account_to_auth(account, auth.CallerID)
                auth.DeepSeekToken = new_auth.DeepSeekToken
                auth.AccountID = new_auth.AccountID
                auth.SourceHeader = new_auth.SourceHeader
                return True
        return False

    def refresh_token(self, auth: RequestAuth) -> bool:
        """Attempt to refresh the token for an account."""
        if not auth.AccountID:
            return False

        account = self.store.account_by_id(auth.AccountID)
        if not account:
            return False

        # If the account has refresh token, try to refresh
        if self.login_func and account.get("refresh_token"):
            try:
                new_token = self.login_func(account)
                if new_token:
                    auth.DeepSeekToken = new_token
                    return True
            except Exception:
                pass

        return False


# Global resolver instance
_resolver: Optional[AuthResolver] = None


def resolve_auth(request, store: Optional[Store] = None) -> RequestAuth:
    """Quick helper to resolve auth from request."""
    global _resolver
    if _resolver is None and store is not None:
        _resolver = AuthResolver(store=store)
    if _resolver is None:
        _resolver = AuthResolver(store=ConfigStore())
    return _resolver.determine(request)
