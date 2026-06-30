"""
Client file status operations
"""

from typing import Dict, Optional

from .client_core import Client
from .errors import DeepSeekError
from deepseek.protocol.constants import DeepSeekFetchFilesURL


def get_file_status(
    client: Client,
    token: str,
    file_id: str,
) -> Dict:
    """
    Get the status of an uploaded file.

    Args:
        client: DeepSeek client
        token: Authentication token
        file_id: File ID to check

    Returns:
        Dict with file status information
    """
    url = f"{DeepSeekFetchFilesURL}/{file_id}"
    data, status, err = client._get_json(
        url,
        client._auth_headers(token),
    )

    if err or status != 200:
        return {
            "error": err or f"HTTP {status}",
            "id": file_id,
            "status": "unknown",
        }

    return data


def list_files(
    client: Client,
    token: str,
    purpose: Optional[str] = None,
) -> Dict:
    """
    List uploaded files.

    Args:
        client: DeepSeek client
        token: Authentication token
        purpose: Optional filter by purpose

    Returns:
        Dict with list of files
    """
    url = DeepSeekFetchFilesURL
    if purpose:
        url = f"{url}?purpose={purpose}"

    data, status, err = client._get_json(
        url,
        client._auth_headers(token),
    )

    if err or status != 200:
        return {"error": err or f"HTTP {status}", "files": []}

    return data
