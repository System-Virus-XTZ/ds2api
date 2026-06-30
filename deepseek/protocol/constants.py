"""
DeepSeek protocol constants

Python port of Go protocol constants for DeepSeek API.
"""

# DeepSeek API host
DeepSeekHost = "chat.deepseek.com"

# API endpoints
DeepSeekLoginURL = f"https://{DeepSeekHost}/api/v0/users/login"
DeepSeekCreateSessionURL = f"https://{DeepSeekHost}/api/v0/chat_session/create"
DeepSeekCreatePowURL = f"https://{DeepSeekHost}/api/v0/chat/create_pow_challenge"
DeepSeekCompletionURL = f"https://{DeepSeekHost}/api/v0/chat/completion"
DeepSeekContinueURL = f"https://{DeepSeekHost}/api/v0/chat/continue"
DeepSeekUploadFileURL = f"https://{DeepSeekHost}/api/v0/files/upload"
DeepSeekFetchFilesURL = f"https://{DeepSeekHost}/api/v0/files"
DeepSeekFetchSessionURL = f"https://{DeepSeekHost}/api/v0/chat_session"
DeepSeekDeleteSessionURL = f"https://{DeepSeekHost}/api/v0/chat_session/delete"
DeepSeekDeleteAllSessionsURL = f"https://{DeepSeekHost}/api/v0/chat_session/delete_all"

# Target paths for PoW
DeepSeekCompletionTargetPath = "/api/v0/chat/completion"
DeepSeekUploadTargetPath = "/api/v0/file/upload_file"

# API version
APIVersion = "v0"

# Client identification
ClientVersion = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
ClientPlatform = "platform/MacIntel"

# Default headers for all requests
BaseHeaders = {
    "User-Agent": ClientVersion,
    "Accept": "application/json, text/event-stream",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": f"https://{DeepSeekHost}",
    "Referer": f"https://{DeepSeekHost}/",
}

# Model identifiers
ModelDeepSeekChat = "deepseek-chat"
ModelDeepSeekReasoner = "deepseek-reasoner"
ModelDeepSeekCoder = "deepseek-coder"

# Model aliases
ModelAliases = {
    "deepseek": ModelDeepSeekChat,
    "deepseek-v3": ModelDeepSeekChat,
    "deepseek-chat": ModelDeepSeekChat,
    "deepseek-r1": ModelDeepSeekReasoner,
    "deepseek-reasoner": ModelDeepSeekReasoner,
    "deepseek-coder": ModelDeepSeekCoder,
    "coder": ModelDeepSeekCoder,
    "code": ModelDeepSeekCoder,
}

# Default model
DefaultModel = ModelDeepSeekChat

# Request limits
MaxRequestBodySize = 100 * 1024 * 1024  # 100 MiB
MaxMessages = 100
MaxTokens = 128000  # Default max tokens
MaxHistoryMessages = 100


# Timeouts
KeepAliveTimeout = 5      # seconds
StreamIdleTimeout = 300   # seconds
MaxKeepaliveCount = 40

def resolve_model(model: str) -> str:
    """Resolve model alias to canonical model name."""
    return ModelAliases.get(model, model)


def is_valid_model(model: str) -> bool:
    """Check if model name is valid or an alias."""
    return model in ModelAliases or model in [
        ModelDeepSeekChat,
        ModelDeepSeekReasoner,
        ModelDeepSeekCoder,
    ]
