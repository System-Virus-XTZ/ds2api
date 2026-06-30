"""
Mobile client configuration
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MobileClientConfig:
    """Mobile client configuration for DeepSeek API."""
    name: str = "DeepSeek"
    platform: str = "android"
    version: str = "3.0.0"
    android_api_level: str = "35"
    locale: str = "zh_CN"

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers: Dict[str, str] = {}

        # Build User-Agent
        user_agent = f"{self.name}/{self.version}"
        if self.platform == "android":
            user_agent += f" Android/{self.android_api_level}"
        headers["User-Agent"] = user_agent

        headers["x-client-platform"] = self.platform
        headers["x-client-version"] = self.version
        headers["x-client-locale"] = self.locale

        return headers


# Default mobile client
DEFAULT_MOBILE_CLIENT = MobileClientConfig()
