"""
SOCKS5 proxy support for DeepSeek client
"""

import socket
import threading
from typing import Callable, Optional, Tuple

import httpx

from config.logger import get_logger

logger = get_logger("proxy")


# Type alias for host lookup function
HostLookupFunc = Callable[[str], Tuple[str, int]]


def proxy_lookup_socks5(proxy_url: str) -> Tuple[str, int]:
    """
    Parse SOCKS5 proxy URL.

    Args:
        proxy_url: SOCKS5 proxy URL (e.g., socks5://127.0.0.1:1080)

    Returns:
        (host, port) tuple
    """
    # Remove protocol prefix
    url = proxy_url
    if url.startswith("socks5://"):
        url = url[len("socks5://"):]
    elif url.startswith("socks5h://"):
        url = url[len("socks5h://"):]

    # Parse host:port
    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            port = 1080
    else:
        host = url
        port = 1080

    return host, port


def socks5_proxy_transport(
    proxy_url: str,
    host_lookup: Optional[HostLookupFunc] = None
):
    """
    Create a SOCKS5 proxy transport for httpx.

    Args:
        proxy_url: SOCKS5 proxy URL
        host_lookup: Optional function to resolve hostnames before connection

    Returns:
        httpx.BaseTransport configured for SOCKS5
    """
    proxy_host, proxy_port = proxy_lookup_socks5(proxy_url)

    # httpx supports SOCKS5 via the Proxy type
    return httpx.HTTPTransport(proxy=httpx.Proxy(f"socks5://{proxy_host}:{proxy_port}"))


def resolve_host_for_proxy(host: str) -> str:
    """
    Resolve hostname to IP address for SOCKS5 proxy compatibility.

    Some SOCKS5 proxies can't resolve hostnames, so we need to
    resolve them ourselves.

    Args:
        host: Hostname to resolve

    Returns:
        IP address string
    """
    try:
        return socket.gethostbyname(host)
    except socket.gaierror as e:
        logger.warning(f"Failed to resolve host {host}: {e}")
        return host


def proxy_request_clients(
    proxy: Optional[str] = None,
    host_lookup: Optional[HostLookupFunc] = None
) -> Tuple[httpx.BaseTransport, httpx.BaseTransport]:
    """
    Create HTTP clients for proxy-aware requests.

    Returns:
        Tuple of (regular_client, stream_client, fallback_client)
    """
    # Regular client (with optional proxy)
    if proxy:
        try:
            transport = socks5_proxy_transport(proxy, host_lookup)
            regular = transport
            stream_transport = socks5_proxy_transport(proxy, host_lookup)
            stream = stream_transport
        except Exception as e:
            logger.warning(f"Failed to create proxy transport: {e}")
            regular = httpx.HTTPTransport()
            stream = httpx.HTTPTransport()
    else:
        regular = httpx.HTTPTransport()
        stream = httpx.HTTPTransport()

    # Fallback client (always direct, no proxy)
    fallback = httpx.HTTPTransport()

    return regular, stream, fallback


class ProxyManager:
    """Manages proxy configuration for multiple accounts."""

    def __init__(self):
        self._lock = threading.Lock()
        self._clients: dict = {}
        self._host_lookup = resolve_host_for_proxy

    def get_client(
        self,
        proxy: Optional[str] = None
    ) -> Tuple[httpx.BaseTransport, httpx.BaseTransport, httpx.BaseTransport]:
        """
        Get HTTP clients for proxy configuration.

        Args:
            proxy: Optional proxy URL

        Returns:
            Tuple of (regular, stream, fallback) transports
        """
        with self._lock:
            if not proxy:
                # Return default non-proxied clients
                return (
                    httpx.HTTPTransport(),
                    httpx.HTTPTransport(),
                    httpx.HTTPTransport(),
                )

            if proxy not in self._clients:
                self._clients[proxy] = proxy_request_clients(
                    proxy,
                    self._host_lookup
                )

            return self._clients[proxy]

    def clear(self) -> None:
        """Clear cached clients."""
        with self._lock:
            self._clients.clear()


# Global proxy manager
_proxy_manager: Optional[ProxyManager] = None


def get_proxy_manager() -> ProxyManager:
    """Get or create global proxy manager."""
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = ProxyManager()
    return _proxy_manager
