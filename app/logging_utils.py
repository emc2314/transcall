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
    if len(text) <= limit:
        return text

    if limit <= 4:
        return f"{text[:limit]}... (truncated, {len(text)} chars total)"

    head_len = max(1, limit // 2)
    tail_len = max(1, limit - head_len)
    head = text[:head_len]
    tail = text[-tail_len:]
    return f"{head}...{tail} (truncated, {len(text)} chars total)"


def truncate_json(value: Any, limit: Optional[int] = None) -> Any:
    """
    Recursively truncate long string values within a JSON-serializable object
    while keeping the overall structure intact.
    """
    max_len = limit if limit is not None else _raw_log_char_limit()

    if isinstance(value, dict):
        return {k: truncate_json(v, max_len) for k, v in value.items()}

    if isinstance(value, list):
        return [truncate_json(item, max_len) for item in value]

    if isinstance(value, tuple):
        return [truncate_json(item, max_len) for item in value]

    if isinstance(value, str):
        if len(value) > max_len:
            return f"{value[:max_len]}... (truncated, {len(value)} chars total)"
        return value

    return value


def format_structured(obj: Any) -> str:
    """
    Produce a truncated JSON string for dict/list/JSON-string inputs
    while preserving structure where possible.
    """
    limit = _raw_log_char_limit()

    def _dump(obj_to_dump: Any) -> Optional[str]:
        try:
            truncated = truncate_json(obj_to_dump, limit)
            return json.dumps(truncated, ensure_ascii=False)
        except Exception:
            return None

    if isinstance(obj, (dict, list, tuple)):
        serialized = _dump(obj)
        if serialized is not None:
            return serialized
        return _truncate(str(obj))

    if isinstance(obj, (bytes, bytearray)):
        try:
            parsed = json.loads(obj.decode("utf-8"))
            serialized = _dump(parsed)
            if serialized is not None:
                return serialized
        except Exception:
            return _truncate(obj.decode("utf-8", errors="replace"))

    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            serialized = _dump(parsed)
            if serialized is not None:
                return serialized
        except json.JSONDecodeError:
            pass
        return _truncate(obj)

    return _truncate(str(obj))


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

    def _dump_truncated_json(obj: Any) -> Optional[str]:
        try:
            truncated = truncate_json(obj)
            return json.dumps(truncated, ensure_ascii=False)
        except Exception:
            return None

    if isinstance(body, bytes):
        decoded_text: Optional[str] = None
        parsed_json: Optional[Any] = None
        try:
            decoded_text = body.decode("utf-8")
            parsed_json = json.loads(decoded_text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed_json = None

        if parsed_json is not None:
            serialized = _dump_truncated_json(parsed_json)
            if serialized is not None:
                return serialized

        if treat_as_binary:
            return format_binary_content(body)

        if decoded_text is None:
            decoded_text = body.decode("utf-8", errors="replace")
        return _truncate(decoded_text)

    if isinstance(body, (dict, list)):
        serialized = _dump_truncated_json(body)
        if serialized is not None:
            return serialized
        return _truncate(str(body))

    if isinstance(body, str):
        parsed_json: Optional[Any] = None
        try:
            parsed_json = json.loads(body)
        except json.JSONDecodeError:
            parsed_json = None

        if parsed_json is not None:
            serialized = _dump_truncated_json(parsed_json)
            if serialized is not None:
                return serialized

        return _truncate(body)

    return _truncate(str(body))


def log_debug_payload(
    logger: logging.Logger,
    prefix: str,
    headers: Optional[Mapping[str, Any]] = None,
    body: Any = None,
    treat_body_as_binary: bool = False,
) -> None:
    """
    Log headers/body payloads when the logger is in DEBUG level.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    header_str = _serialize_headers(headers)
    body_str = _serialize_body(body, treat_body_as_binary)
    logger.debug("%s | headers=%s | body=%s", prefix, header_str, body_str)
