"""
Client session management
"""

from typing import Optional

from .client_core import Client, SessionResult
from .errors import DeepSeekError


def create_session(
    client: Client,
    token: str,
    max_attempts: int = 3,
) -> str:
    """
    Create a new DeepSeek chat session.

    Args:
        client: DeepSeek client
        token: Authentication token
        max_attempts: Maximum retry attempts

    Returns:
        Session ID

    Raises:
        DeepSeekError: If session creation fails
    """
    return client.create_session(token, max_attempts)


def delete_session(
    client: Client,
    token: str,
    session_id: str,
) -> SessionResult:
    """
    Delete a DeepSeek chat session.

    Args:
        client: DeepSeek client
        token: Authentication token
        session_id: Session to delete

    Returns:
        DeleteSessionResult
    """
    return client.delete_session(token, session_id)


def delete_all_sessions(
    client: Client,
    token: str,
) -> bool:
    """
    Delete all DeepSeek chat sessions.

    Args:
        client: DeepSeek client
        token: Authentication token

    Returns:
        True if successful
    """
    return client.delete_all_sessions(token)
