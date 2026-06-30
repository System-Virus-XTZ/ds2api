#!/usr/bin/env python3
"""
ds2api - DeepSeek API Proxy with Account Pooling

A production-ready API proxy that provides OpenAI-compatible endpoints
for DeepSeek's chat API, with multi-account pooling and rate limit handling.

Usage:
    python main.py [--config CONFIG_PATH] [--host HOST] [--port PORT]

Environment variables:
    CONFIG_PATH          Path to config.json (default: config.json)
    DS2API_CONFIG_PATH  Alternative config path
    HOST                 Listen host (default: 0.0.0.0)
    PORT                 Listen port (default: 8080)
    DEBUG                Enable debug mode (default: false)
    LOG_LEVEL            Log level (default: INFO)
"""

import argparse
import asyncio
import logging
import os
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        prog="ds2api",
        description="DeepSeek API Proxy with account pooling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        dest="config",
        default=os.environ.get("DS2API_CONFIG_PATH", os.environ.get("CONFIG_PATH", "config.json")),
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Listen host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Listen port (default: 8080)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Bootstrap config if needed
    config_path = args.config
    if not os.path.exists(config_path):
        _create_example_config(config_path)

    # Pass config path via environment variable (compatible with newer uvicorn)
    os.environ["DS2API_CONFIG_PATH"] = config_path

    # Import after config check
    from config.logger import setup_logging
    setup_logging()

    # Start server
    import logging
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    uvicorn.run(
        "server.router:get_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
    )


def _create_example_config(path: str):
    """Create example config if it doesn't exist."""
    example = {
        "bind": "0.0.0.0:8000",
        "debug": False,
        "log_level": "info",
        "accounts": [],
        "api_keys": [],
        "managed_accounts": {
            "enabled": True,
            "default_max_inflight": 3,
        },
        "global_max_inflight": 10,
        "account_max_inflight": 5,
        "pool_cycle_seconds": 300,
        "chat_history": {
            "path": ".chat_history.json",
            "limit": 20,
        },
        "server": {
            "request_timeout": 60,
            "read_timeout": 60,
            "write_timeout": 600,
        },
        "jwt": {
            "secret": "change-me-in-production",
            "expire_hours": 24,
        },
        "auto_delete_sessions": "none",
        "allowed_origins": ["*"],
    }
    import json
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(example, f, indent=2)
    print(f"Created example config: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
