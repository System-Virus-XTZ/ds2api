"""
Logger configuration for ds2api.

Provides both stdlib logging integration and a convenience Logger class
with Info/Warn/Debug/Error class methods.
"""

import logging
import os
import sys
from typing import Optional


# ─── Internal stdlib logger ───────────────────────────────────────────────────

_global_logger: Optional[logging.Logger] = None
_is_configured = False


def _ensure_configured():
    global _global_logger, _is_configured
    if not _is_configured:
        setup_logging()
    return _global_logger


# ─── Public API ───────────────────────────────────────────────────────────────


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configure the root ds2api logger.
    Called once at startup.
    """
    global _global_logger, _is_configured

    if _is_configured:
        return _global_logger or logging.getLogger("ds2api")

    log_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    logger = logging.getLogger("ds2api")
    logger.setLevel(numeric_level)

    # Remove existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Console handler with colored output
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric_level)

    # Format
    RESET = "\033[0m"
    COLOR_MAP = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",    # Green
        "WARNING": "\033[33m", # Yellow
        "ERROR": "\033[31m",   # Red
    }
    color = COLOR_MAP.get(log_level, "")

    formatter = logging.Formatter(
        f"%(asctime)s {color}%(levelname)s{RESET} %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    _global_logger = logger
    _is_configured = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger for a submodule."""
    if name:
        return logging.getLogger(f"ds2api.{name}")
    return _ensure_configured()


def configure_logger(level: Optional[str] = None) -> logging.Logger:
    """Alias for setup_logging."""
    return setup_logging(level)


# ─── Logger wrapper class ──────────────────────────────────────────────────────
# Provides Logger.Info(...) / Logger.Warn(...) / Logger.Error(...) class methods
# that delegate to the configured root logger.


class Logger:
    """
    Convenience wrapper providing class-method logging.

    Usage:
        Logger.Info("message")   # class method
        Logger.Warn("message")   # class method
    """
    _logger: Optional[logging.Logger] = None

    @classmethod
    def _get(cls) -> logging.Logger:
        if cls._logger is None:
            cls._logger = _ensure_configured()
        return cls._logger

    @classmethod
    def Info(cls, msg: str, **kwargs):
        if kwargs:
            extra = " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
            msg = msg + extra
        cls._get().info(msg)

    @classmethod
    def Warn(cls, msg: str, **kwargs):
        if kwargs:
            extra = " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
            msg = msg + extra
        cls._get().warning(msg)

    @classmethod
    def Debug(cls, msg: str, **kwargs):
        if kwargs:
            extra = " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
            msg = msg + extra
        cls._get().debug(msg)

    @classmethod
    def Error(cls, msg: str, **kwargs):
        if kwargs:
            extra = " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
            msg = msg + extra
        cls._get().error(msg)

    @classmethod
    def Fatal(cls, msg: str, **kwargs):
        if kwargs:
            extra = " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
            msg = msg + extra
        cls._get().critical(msg)


# ─── LazyLogger ───────────────────────────────────────────────────────────────


class LazyLogger:
    """Lazy logger that configures on first use."""

    def __init__(self, name: str):
        self._name = name
        self._logger: Optional[logging.Logger] = None

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = get_logger(self._name)
        return self._logger

    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, **kwargs)

    def warn(self, msg: str, **kwargs):
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, **kwargs)

    def exception(self, msg: str, **kwargs):
        self.logger.exception(msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self.logger.critical(msg, **kwargs)
