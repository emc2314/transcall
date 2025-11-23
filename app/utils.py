from typing import Optional

def detect_content_type(data: bytes) -> Optional[str]:
    """
    Detects MIME type based on file signature (magic numbers).
    Supports common formats used in AI image generation: PNG, JPEG, WEBP.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None
