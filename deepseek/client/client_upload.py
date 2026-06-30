"""
Client file upload operations
"""

from typing import Optional

from .client_core import Client, UploadFileResult
from .errors import DeepSeekError


def upload_file(
    client: Client,
    token: str,
    file_data: bytes,
    filename: str,
    purpose: str = "chat",
    max_attempts: int = 3,
) -> UploadFileResult:
    """
    Upload a file to DeepSeek.

    Args:
        client: DeepSeek client
        token: Authentication token
        file_data: File content as bytes
        filename: File name
        purpose: Purpose of upload (default: chat)
        max_attempts: Maximum retry attempts

    Returns:
        UploadFileResult with file_id on success
    """
    return client.upload_file(token, file_data, filename, max_attempts)


def upload_file_from_path(
    client: Client,
    token: str,
    file_path: str,
    purpose: str = "chat",
) -> UploadFileResult:
    """
    Upload a file from filesystem path.

    Args:
        client: DeepSeek client
        token: Authentication token
        file_path: Path to file
        purpose: Purpose of upload

    Returns:
        UploadFileResult
    """
    import os

    filename = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        file_data = f.read()

    return upload_file(client, token, file_data, filename, purpose)


def get_file_status(
    client: Client,
    token: str,
    file_id: str,
) -> dict:
    """
    Get status of an uploaded file.

    Args:
        client: DeepSeek client
        token: Authentication token
        file_id: File ID

    Returns:
        File status dict
    """
    return client.get_file_status(token, file_id)
