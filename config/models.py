"""
Config models - data structures for ds2api configuration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ModelConfig:
    """Model configuration for a DeepSeek model."""
    name: str
    model: str
    enabled: bool = True
    hidden: bool = False
    disabled_reason: str = ""
    max_tokens: int = 8192
    supports_thinking: bool = True
    supports_search: bool = False
    supports_tools: bool = True


@dataclass
class ModelAlias:
    """Model alias mapping."""
    alias: str
    target: str


@dataclass
class SupportedModel:
    """Supported model with config and alias info."""
    config: ModelConfig
    is_alias: bool = False


# Default supported models
DEFAULT_MODELS: List[ModelConfig] = [
    ModelConfig(name="deepseek-chat", model="deepseek-chat"),
    ModelConfig(name="deepseek-reasoner", model="deepseek-reasoner", supports_thinking=True),
    ModelConfig(name="deepseek-coder", model="deepseek-coder", supports_tools=True),
]

# Model aliases
MODEL_ALIASES: Dict[str, str] = {
    "deepseek": "deepseek-chat",
    "deepseek-v3": "deepseek-chat",
    "deepseek-r1": "deepseek-reasoner",
    "coder": "deepseek-coder",
    "code": "deepseek-coder",
}

# Default config
DEFAULT_CONFIG = {
    "log_level": "info",
    "port": 8080,
    "host": "0.0.0.0",
    "model_aliases": MODEL_ALIASES,
    "models": [m.__dict__ for m in DEFAULT_MODELS],
}
