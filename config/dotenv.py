"""
Dotenv loader - load configuration from .env file
"""

import os
from typing import Dict, Optional


def load_dotenv(path: Optional[str] = None) -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_path = path or os.path.join(os.getcwd(), ".env")
    env_vars: Dict[str, str] = {}

    if not os.path.exists(env_path):
        return env_vars

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                # Don't override existing environment variables
                if key not in os.environ:
                    os.environ[key] = value
                    env_vars[key] = value

    return env_vars


def get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


def get_env_int(key: str, default: int = 0) -> int:
    """Get environment variable as integer."""
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean."""
    val = os.environ.get(key, "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    return default
