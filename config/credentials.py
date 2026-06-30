"""
Credentials configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CredentialsConfig:
    """API credentials configuration."""
    admin_key: str = ""
    jwt_secret: str = "change-me-in-production"
    jwt_expiry_hours: int = 24

    def validate(self) -> bool:
        """Validate credentials configuration."""
        if not self.admin_key:
            return False
        if len(self.jwt_secret) < 8:
            return False
        return True
