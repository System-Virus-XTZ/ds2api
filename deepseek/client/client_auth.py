"""
Client authentication methods
"""

from typing import Dict, Any

from .client_core import Client, LoginResult
from .errors import DeepSeekError


def login_account(client: Client, account: Dict[str, Any]) -> str:
    """
    Login to DeepSeek using account credentials.

    Args:
        client: DeepSeek client
        account: Account dict with credentials

    Returns:
        Authentication token

    Raises:
        DeepSeekError: If login fails
    """
    email = account.get("email", "")
    phone = account.get("phone", "")
    password = account.get("password", "")
    token = account.get("token", "")

    # If account already has token, use it
    if token:
        return token

    # Otherwise, login
    if email and password:
        result = client.login_email(email, password)
        if result.success:
            return result.token
        raise DeepSeekError(f"Email login failed: {result.error}")

    if phone:
        # Phone login requires verification code
        raise DeepSeekError("Phone login requires verification code")

    raise DeepSeekError("No credentials available")


def refresh_auth_token(client: Client, refresh_token: str) -> str:
    """
    Refresh authentication token.

    Args:
        client: DeepSeek client
        refresh_token: Current refresh token

    Returns:
        New authentication token

    Raises:
        DeepSeekError: If refresh fails
    """
    new_token = client.refresh_token(refresh_token)
    if new_token:
        return new_token
    raise DeepSeekError("Token refresh not available")
