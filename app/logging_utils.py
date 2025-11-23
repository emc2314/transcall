import base64
import json
import logging
import os
from typing import Any, Mapping, Optional


def _raw_log_char_limit() -> int:
    return int(os.getenv("RAW_LOG_CHAR_LIMIT", "4096"))


def _binary_mode() -> str:
    return os.getenv("RAW_LOG_BINARY_MODE", "summary").strip().lower()


def _truncate(text: str) -> str:
    limit = _raw_log_char_limit()
    if len(text) > limit:
        return f"{text[:limit]}... (truncated, {len(text)} chars total)"
    return text


def _serialize_headers(headers: Optional[Mapping[str, Any]]) -> str:
    if headers is None:
        return ""
    
    sensitive_keys = {"authorization", "api-key", "x-api-key", "cookie", "token"}
    
    try:
        header_dict = {}
        for k, v in headers.items():
            key_lower = str(k).lower()
            if key_lower in sensitive_keys:
                val_str = str(v)
                if len(val_str) > 4:
                    header_dict[k] = f"{val_str[:2]}***{val_str[-2:]}"
                else:
                    header_dict[k] = "***"
            else:
                header_dict[k] = str(v)
    except Exception:
        # Fallback if iteration fails, though unlikely with Mapping
        header_dict = {"error": "Failed to serialize headers"}
        
    return _truncate(json.dumps(header_dict, ensure_ascii=False))


def format_binary_content(data: bytes) -> str:
    mode = _binary_mode()
    if mode == "raw":
        text = data.decode("latin-1", errors="replace")
        return _truncate(text)
    if mode == "base64":
        b64 = base64.b64encode(data).decode("ascii")
        return _truncate(b64)
    return f"<binary {len(data)} bytes>"


def _serialize_body(body: Any, treat_as_binary: bool = False) -> str:
    if body is None:
        return ""
    if isinstance(body, bytes):
        if treat_as_binary:
            return format_binary_content(body)
        text = body.decode("utf-8", errors="replace")
        return _truncate(text)
    if isinstance(body, (dict, list)):
        try:
            text = json.dumps(body, ensure_ascii=False)
        except Exception:
            text = str(body)
        return _truncate(text)
    return _truncate(str(body))


def log_debug_payload(
    logger: logging.Logger,
    prefix: str,
    headers: Optional[Mapping[str, Any]] = None,
    body: Any = None,
    treat_body_as_binary: bool = False
) -> None:
    """
    Log headers/body payloads when the logger is in DEBUG level.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    header_str = _serialize_headers(headers)
    body_str = _serialize_body(body, treat_body_as_binary)
    logger.debug("%s | headers=%s | body=%s", prefix, header_str, body_str)
