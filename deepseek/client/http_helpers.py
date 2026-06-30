"""
HTTP helpers for DeepSeek client

Utility functions for HTTP request/response handling including
compression decompression and header manipulation.
"""

import gzip
import io
from typing import Dict, Optional

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

from config.logger import get_logger

logger = get_logger("http_helpers")


def read_response_body(response) -> bytes:
    """
    Read and decompress response body.

    Handles gzip and brotli compression.

    Args:
        response: httpx.Response object

    Returns:
        Decompressed body bytes
    """
    encoding = response.headers.get("Content-Encoding", "").lower().strip()

    body = response.read()

    if encoding == "gzip":
        try:
            body = decompress_gzip(body)
        except Exception as e:
            logger.warning(f"Failed to decompress gzip: {e}")

    elif encoding == "br" and HAS_BROTLI:
        try:
            body = brotli.decompress(body)
        except Exception as e:
            logger.warning(f"Failed to decompress brotli: {e}")

    return body


def decompress_gzip(data: bytes) -> bytes:
    """Decompress gzip data."""
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        return f.read()


def preview(data: bytes, max_len: int = 160) -> str:
    """Get a preview of response body for logging."""
    s = data.decode("utf-8", errors="replace").strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def clone_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Create a copy of headers dict."""
    return dict(headers)


def json_headers(headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Get headers with JSON Content-Type."""
    out = clone_headers(headers) if headers else {}
    out["Content-Type"] = "application/json"
    return out


def merge_headers(base: Dict[str, str], override: Dict[str, str]) -> Dict[str, str]:
    """Merge two header dicts."""
    result = dict(base)
    result.update(override)
    return result
