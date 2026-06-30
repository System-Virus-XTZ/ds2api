"""
API index
"""

# Re-export from server
from server.router import DS2APIServer

router = DS2APIServer

__all__ = ["DS2APIServer", "router"]
