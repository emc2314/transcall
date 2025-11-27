import logging
from typing import Tuple, Any, Dict

from fastapi.responses import JSONResponse

from app import exceptions as exc

logger = logging.getLogger("ErrorHandler")

_UPSTREAM_HINTS = [
    ("safety", 400, "UPSTREAM_SAFETY_BLOCKED"),
    ("policy", 400, "UPSTREAM_POLICY_BLOCKED"),
    ("sensitive", 400, "UPSTREAM_POLICY_BLOCKED"),
    ("rate limit", 429, "UPSTREAM_RATE_LIMIT"),
    ("quota", 429, "UPSTREAM_RATE_LIMIT"),
    ("too many requests", 429, "UPSTREAM_RATE_LIMIT"),
    ("overloaded", 503, "UPSTREAM_OVERLOADED"),
    ("unavailable", 503, "UPSTREAM_UNAVAILABLE"),
]


def _classify_upstream_error(err: exc.UpstreamResponseError) -> Tuple[int, str]:
    message = err.detail.lower()
    for needle, status, code in _UPSTREAM_HINTS:
        if needle in message:
            return status, code
    if 400 <= err.upstream_status < 600:
        return err.upstream_status, "UPSTREAM_HTTP_ERROR"
    return err.status_code, err.error_code


async def transcall_exception_handler(request, err: exc.TranscallError):
    status_code = err.status_code
    error_code = err.error_code
    meta = dict(err.meta or {})

    if isinstance(err, exc.UpstreamResponseError):
        status_code, error_code = _classify_upstream_error(err)

    payload: Dict[str, Any] = {"error": error_code, "detail": err.detail}
    if meta:
        payload["meta"] = meta

    logger.debug(
        "Handled TranscallError error_code=%s status=%s meta=%s",
        error_code,
        status_code,
        meta or {},
    )

    return JSONResponse(status_code=status_code, content=payload)
