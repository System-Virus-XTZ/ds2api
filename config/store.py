"""
ConfigStore - Central configuration management for ds2api

This is the Python port of the Go ConfigStore that loads and manages
all configuration from JSON files and environment variables.
"""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .logger import get_logger, configure_logger

logger = get_logger("config")


@dataclass
class Store:
    """
    Central configuration store for ds2api.

    Manages accounts, API keys, model aliases, and all runtime settings.
    Thread-safe for concurrent reads.
    """
    # Raw config data
    _data: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # Cached values (populated lazily)
    _accounts_cache: Optional[List[Dict]] = field(default=None, repr=False)
    _keys_cache: Optional[List[str]] = field(default=None, repr=False)
    _model_aliases_cache: Optional[Dict[str, str]] = field(default=None, repr=False)
    _models_cache: Optional[List[Dict]] = field(default=None, repr=False)
    _proxies_cache: Optional[List[str]] = field(default=None, repr=False)

    def __post_init__(self):
        if not self._data:
            self._data = self._load_defaults()

    @staticmethod
    def _load_defaults() -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "log_level": "info",
            "port": 8080,
            "host": "0.0.0.0",
            "global_max_inflight": 10,
            "account_max_inflight": 3,
            "wait_timeout": 300,
            "cleanup_interval": 60,
            "auto_delete_sessions": "none",  # "none", "single", "all"
            "toolcall_mode": "auto",
            "toolcall_early_emit_confidence": "medium",
            "responses_store_ttl_seconds": 86400,
            "embeddings_provider": "deepseek",
            "current_input_file_enabled": True,
            "current_input_file_min_chars": 1000,
            "thinking_injection_enabled": True,
            "thinking_injection_prompt": "",
            "chat_history_path": "",
            "accounts": [],
            "keys": [],
            "model_aliases": {},
            "models": [],
            "proxies": [],
        }

    @property
    def data(self) -> Dict[str, Any]:
        """Get raw config data (read-only copy)."""
        with self._lock:
            return dict(self._data)

    def _get(self, key: str, default: Any = None) -> Any:
        """Get config value by key."""
        with self._lock:
            return self._data.get(key, default)

    def _set(self, key: str, value: Any) -> None:
        """Set config value by key."""
        with self._lock:
            self._data[key] = value
            # Invalidate caches
            self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate all caches."""
        self._accounts_cache = None
        self._keys_cache = None
        self._model_aliases_cache = None
        self._models_cache = None
        self._proxies_cache = None

    # === Account Management ===

    def accounts(self) -> List[Dict[str, Any]]:
        """Get all configured accounts."""
        if self._accounts_cache is not None:
            return self._accounts_cache
        with self._lock:
            self._accounts_cache = self._data.get("accounts", [])
            return list(self._accounts_cache)

    def account_by_id(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Find account by ID."""
        for acc in self.accounts():
            if acc.get("id") == account_id:
                return acc
        return None

    def account_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find account by email."""
        for acc in self.accounts():
            if acc.get("email", "").lower() == email.lower():
                return acc
        return None

    def account_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """Find account by phone."""
        for acc in self.accounts():
            if acc.get("phone") == phone:
                return acc
        return None

    # === API Keys ===

    def keys(self) -> List[str]:
        """Get all configured API keys."""
        if self._keys_cache is not None:
            return self._keys_cache
        with self._lock:
            self._keys_cache = self._data.get("keys", [])
            return list(self._keys_cache)

    def key_exists(self, key: str) -> bool:
        """Check if an API key exists."""
        return key in self.keys()

    # === Model Configuration ===

    def models(self) -> List[Dict[str, Any]]:
        """Get configured models."""
        if self._models_cache is not None:
            return self._models_cache
        with self._lock:
            self._models_cache = self._data.get("models", [])
            return list(self._models_cache)

    def model_aliases(self) -> Dict[str, str]:
        """Get model alias mappings."""
        if self._model_aliases_cache is not None:
            return self._model_aliases_cache
        with self._lock:
            self._model_aliases_cache = self._data.get("model_aliases", {})
            return dict(self._model_aliases_cache)

    def resolve_model(self, model: str) -> str:
        """Resolve model alias to actual model name."""
        aliases = self.model_aliases()
        return aliases.get(model, model)

    def model_enabled(self, model: str) -> bool:
        """Check if model is enabled."""
        model = self.resolve_model(model)
        for m in self.models():
            if m.get("model") == model or m.get("name") == model:
                return m.get("enabled", True)
        return True  # Unknown models are allowed by default

    def get_model_config(self, model: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        model = self.resolve_model(model)
        for m in self.models():
            if m.get("model") == model or m.get("name") == model:
                return m
        return None

    # === Proxy Configuration ===

    def proxies(self) -> List[str]:
        """Get configured proxy list."""
        if self._proxies_cache is not None:
            return self._proxies_cache
        with self._lock:
            self._proxies_cache = self._data.get("proxies", [])
            return list(self._proxies_cache)

    def proxy_for_account(self, account_id: str) -> Optional[str]:
        """Get proxy for a specific account."""
        account = self.account_by_id(account_id)
        if account and account.get("proxy"):
            return account["proxy"]
        proxies = self.proxies()
        if proxies:
            # Round-robin or first proxy
            return proxies[0]
        return None

    # === Rate Limiting ===

    def global_max_inflight(self) -> int:
        """Get global maximum concurrent requests."""
        return self._get("global_max_inflight", 10)

    def account_max_inflight(self) -> int:
        """Get per-account maximum concurrent requests."""
        return self._get("account_max_inflight", 3)

    def wait_timeout(self) -> int:
        """Get wait timeout in seconds."""
        return self._get("wait_timeout", 300)

    def cleanup_interval(self) -> int:
        """Get cleanup interval in seconds."""
        return self._get("cleanup_interval", 60)

    # === Feature Flags ===

    def toolcall_mode(self) -> str:
        """Get tool call mode: auto, force, none."""
        return self._get("toolcall_mode", "auto")

    def toolcall_early_emit_confidence(self) -> str:
        """Get tool call early emit confidence: high, medium, low."""
        return self._get("toolcall_early_emit_confidence", "medium")

    def responses_store_ttl_seconds(self) -> int:
        """Get responses store TTL in seconds."""
        return self._get("responses_store_ttl_seconds", 86400)

    def embeddings_provider(self) -> str:
        """Get embeddings provider."""
        return self._get("embeddings_provider", "deepseek")

    def auto_delete_mode(self) -> str:
        """Get auto delete sessions mode: none, single, all."""
        return self._get("auto_delete_sessions", "none")

    def auto_delete_sessions(self) -> bool:
        """Check if auto delete sessions is enabled."""
        return self.auto_delete_mode() != "none"

    def current_input_file_enabled(self) -> bool:
        """Check if current input file feature is enabled."""
        return self._get("current_input_file_enabled", True)

    def current_input_file_min_chars(self) -> int:
        """Get minimum characters for current input file."""
        return self._get("current_input_file_min_chars", 1000)

    def thinking_injection_enabled(self) -> bool:
        """Check if thinking injection is enabled."""
        return self._get("thinking_injection_enabled", True)

    def thinking_injection_prompt(self) -> str:
        """Get thinking injection prompt."""
        return self._get("thinking_injection_prompt", "")

    # === Chat History ===

    def chat_history_path(self) -> str:
        """Get chat history storage path."""
        return self._get("chat_history_path", "")

    # === Admin / Security ===

    def admin_key(self) -> str:
        """Get admin API key."""
        return self._get("admin_key", "")

    def jwt_secret(self) -> str:
        """Get JWT signing secret."""
        return self._get("jwt_secret", "change-me-in-production")

    def jwt_expiry_hours(self) -> int:
        """Get JWT expiry in hours."""
        return self._get("jwt_expiry_hours", 24)

    # === Server Configuration ===

    def log_level(self) -> str:
        """Get log level."""
        return self._get("log_level", "info")

    def port(self) -> int:
        """Get server port."""
        return self._get("port", 8080)

    def host(self) -> str:
        """Get server host."""
        return self._get("host", "0.0.0.0")

    # === Update Methods ===

    def update_account(self, account_id: str, updates: Dict[str, Any]) -> bool:
        """Update account configuration."""
        with self._lock:
            for i, acc in enumerate(self._data.get("accounts", [])):
                if acc.get("id") == account_id:
                    self._data["accounts"][i].update(updates)
                    self._invalidate_cache()
                    return True
            return False

    def add_account(self, account: Dict[str, Any]) -> bool:
        """Add a new account."""
        with self._lock:
            if "accounts" not in self._data:
                self._data["accounts"] = []
            self._data["accounts"].append(account)
            self._invalidate_cache()
            return True

    def remove_account(self, account_id: str) -> bool:
        """Remove an account."""
        with self._lock:
            accounts = self._data.get("accounts", [])
            for i, acc in enumerate(accounts):
                if acc.get("id") == account_id:
                    self._data["accounts"].pop(i)
                    self._invalidate_cache()
                    return True
            return False

    def add_key(self, key: str) -> bool:
        """Add an API key."""
        with self._lock:
            if "keys" not in self._data:
                self._data["keys"] = []
            if key not in self._data["keys"]:
                self._data["keys"].append(key)
                self._invalidate_cache()
                return True
            return False

    def remove_key(self, key: str) -> bool:
        """Remove an API key."""
        with self._lock:
            keys = self._data.get("keys", [])
            if key in keys:
                self._data["keys"].remove(key)
                self._invalidate_cache()
                return True
            return False

    def set_proxy(self, proxy: str, account_id: Optional[str] = None) -> bool:
        """Set proxy for account or globally."""
        with self._lock:
            if account_id:
                return self.update_account(account_id, {"proxy": proxy})
            if "proxies" not in self._data:
                self._data["proxies"] = []
            if proxy not in self._data["proxies"]:
                self._data["proxies"].append(proxy)
                self._invalidate_cache()
            return True

    def remove_proxy(self, proxy: str) -> bool:
        """Remove a proxy."""
        with self._lock:
            proxies = self._data.get("proxies", [])
            if proxy in proxies:
                self._data["proxies"].remove(proxy)
                self._invalidate_cache()
                return True
            return False


# === Global Store Instance ===

_global_store: Optional[Store] = None
_store_lock = threading.Lock()


def ConfigStore() -> Store:
    """Get or create the global configuration store."""
    global _global_store
    with _store_lock:
        if _global_store is None:
            _global_store = Store()
        return _global_store


def LoadStoreWithError() -> Store:
    """Load configuration from file and environment variables."""
    global _global_store

    configure_logger()

    # Try to load from environment
    config_path = os.environ.get("DS2API_CONFIG_PATH", "")
    config_json = os.environ.get("DS2API_CONFIG_JSON", "")

    # Try default paths
    default_paths = [
        config_path or "config.json",
        os.path.expanduser("~/.config/ds2api/config.json"),
        "/etc/ds2api/config.json",
    ]

    config_data: Dict[str, Any] = {}

    # Load from JSON string
    if config_json:
        try:
            config_data = json.loads(config_json)
            logger.info("Loaded config from DS2API_CONFIG_JSON")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse DS2API_CONFIG_JSON: {e}")

    # Load from file
    if not config_data:
        for path in default_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        config_data = json.load(f)
                    logger.info(f"Loaded config from {path}")
                    break
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load config from {path}: {e}")

    # Apply environment variable overrides
    env_overrides = _load_env_overrides()
    config_data.update(env_overrides)

    # Create store
    with _store_lock:
        _global_store = Store(_data=config_data)
        logger.info(
            f"Config loaded: {len(config_data.get('accounts', []))} accounts, "
            f"{len(config_data.get('keys', []))} keys"
        )

    return _global_store


def _load_env_overrides() -> Dict[str, Any]:
    """Load configuration overrides from environment variables."""
    overrides: Dict[str, Any] = {}

    # Admin settings
    if admin_key := os.environ.get("DS2API_ADMIN_KEY"):
        overrides["admin_key"] = admin_key

    if jwt_secret := os.environ.get("DS2API_JWT_SECRET"):
        overrides["jwt_secret"] = jwt_secret

    # Chat history
    if chat_history_path := os.environ.get("DS2API_CHAT_HISTORY_PATH"):
        overrides["chat_history_path"] = chat_history_path

    # Server settings
    if port := os.environ.get("PORT"):
        try:
            overrides["port"] = int(port)
        except ValueError:
            pass

    if log_level := os.environ.get("LOG_LEVEL"):
        overrides["log_level"] = log_level.lower()

    # Vercel deployment
    if os.environ.get("VERCEL"):
        overrides["port"] = 8080
        overrides["host"] = "0.0.0.0"

    return overrides


def ResetStore() -> None:
    """Reset the global store (for testing)."""
    global _global_store
    with _store_lock:
        _global_store = None
