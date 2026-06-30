"""
Config validation utilities
"""

import re
from typing import List, Optional, Tuple


class ValidationError(Exception):
    """Configuration validation error."""

    def __init__(self, message: str, field: str = ""):
        self.message = message
        self.field = field
        super().__init__(message)


def validate_email(email: str) -> bool:
    """Validate email format."""
    if not email:
        return False
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    if not phone:
        return False
    # Allow international format with + prefix
    pattern = r"^\+?[1-9]\d{6,14}$"
    return bool(re.match(pattern, phone.replace(" ", "").replace("-", "")))


def validate_proxy(proxy: str) -> bool:
    """Validate proxy URL format."""
    if not proxy:
        return True  # Empty is OK (no proxy)
    patterns = [
        r"^socks5://.+",
        r"^socks5h://.+",
        r"^http://.+",
        r"^https://.+",
    ]
    return any(re.match(p, proxy) for p in patterns)


def validate_account(account_data: dict) -> List[str]:
    """Validate account configuration and return list of errors."""
    errors: List[str] = []

    if not account_data.get("id"):
        errors.append("account id is required")

    email = account_data.get("email", "")
    phone = account_data.get("phone", "")
    token = account_data.get("token", "")

    if not email and not phone and not token:
        errors.append("account must have email, phone, or token")

    if email and not validate_email(email):
        errors.append(f"invalid email format: {email}")

    if phone and not validate_phone(phone):
        errors.append(f"invalid phone format: {phone}")

    proxy = account_data.get("proxy", "")
    if proxy and not validate_proxy(proxy):
        errors.append(f"invalid proxy format: {proxy}")

    max_inflight = account_data.get("max_inflight", 3)
    if max_inflight < 1:
        errors.append("max_inflight must be at least 1")

    return errors


def validate_config(config: dict) -> List[str]:
    """Validate entire config and return list of errors."""
    errors: List[str] = []

    # Validate admin key
    admin_key = config.get("admin_key", "")
    if not admin_key:
        errors.append("admin_key is required for security")

    # Validate JWT secret
    jwt_secret = config.get("jwt_secret", "change-me-in-production")
    if len(jwt_secret) < 8:
        errors.append("jwt_secret must be at least 8 characters")

    # Validate accounts
    accounts = config.get("accounts", [])
    for i, account in enumerate(accounts):
        account_errors = validate_account(account)
        for err in account_errors:
            errors.append(f"account[{i}]: {err}")

    # Validate model aliases
    aliases = config.get("model_aliases", {})
    for alias, target in aliases.items():
        if not target:
            errors.append(f"model alias '{alias}' has empty target")

    # Validate rate limits
    global_max = config.get("global_max_inflight", 10)
    if global_max < 1:
        errors.append("global_max_inflight must be at least 1")

    return errors
