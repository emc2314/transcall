from typing import Any, Dict, Optional


class TranscallError(Exception):
    status_code: int = 500
    error_code: str = "TRANSCALL_ERROR"

    def __init__(self, detail: str, *, meta: Optional[Dict[str, Any]] = None):
        super().__init__(detail)
        self.detail = detail
        self.meta = meta or {}


class ModelNotSupportedError(TranscallError):
    status_code = 400
    error_code = "MODEL_NOT_SUPPORTED"

    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' is not configured for this service.",
            meta={"model": model_name},
        )


class ProviderNotRegisteredError(TranscallError):
    status_code = 500
    error_code = "PROVIDER_NOT_REGISTERED"

    def __init__(self, provider_name: str):
        super().__init__(
            f"Provider '{provider_name}' is not registered.",
            meta={"provider": provider_name},
        )


class InvalidModelConfigError(TranscallError):
    status_code = 500
    error_code = "INVALID_MODEL_CONFIG"


class ProviderAuthenticationError(TranscallError):
    status_code = 500
    error_code = "UPSTREAM_AUTH_ERROR"


class UpstreamTimeoutError(TranscallError):
    status_code = 504
    error_code = "UPSTREAM_TIMEOUT"

    def __init__(self, provider: str, endpoint: str):
        super().__init__(
            "Upstream service timed out.",
            meta={"provider": provider, "endpoint": endpoint},
        )


class UpstreamConnectionError(TranscallError):
    status_code = 502
    error_code = "UPSTREAM_CONNECTION_ERROR"

    def __init__(self, provider: str, message: str, endpoint: str):
        super().__init__(
            f"Failed to connect to upstream provider: {message}",
            meta={"provider": provider, "endpoint": endpoint},
        )


class UpstreamResponseError(TranscallError):
    status_code = 502
    error_code = "UPSTREAM_RESPONSE_ERROR"

    def __init__(
        self,
        provider: str,
        upstream_status: int,
        message: str,
        *,
        payload: Optional[Any] = None,
    ):
        meta = {
            "provider": provider,
            "upstream_status": upstream_status,
        }
        if payload is not None:
            meta["upstream_payload"] = payload
        super().__init__(message, meta=meta)
        self.provider = provider
        self.upstream_status = upstream_status
        self.payload = payload


class InternalServiceError(TranscallError):
    status_code = 500
    error_code = "INTERNAL_SERVICE_ERROR"
