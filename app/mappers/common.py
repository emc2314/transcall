import base64
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from app.schemas import UnifiedMessage, UnifiedContentPart
from app.utils import detect_content_type

logger = logging.getLogger("Mappers")

OPENAI_OPTIONAL_FIELDS: Set[str] = {
    "n",
    "size",
    "response_format",
    "style",
    "background",
    "moderation",
    "quality",
    "output_format",
    "output_compression",
    "partial_images",
    "stream",
    "user",
    "input_fidelity",
}

GEMINI_OPTIONAL_FIELDS: Set[str] = {
    "generationConfig",
    "safetySettings",
    "tools",
    "toolConfig",
    "systemInstruction",
    "cachedContent",
    "responseModalities",
}

OPENAI_ALLOWED_SIZES: Set[str] = {"1024x1024", "1536x1024", "1024x1536"}


def warn_unknown_fields(data: Dict[str, Any], known_fields: Set[str], context: str):
    unknown = set(data.keys()) - known_fields
    if unknown:
        logger.warning("[%s] Unknown fields found and ignored: %s", context, unknown)


def should_include_openai_field(req, field_name: str) -> bool:
    if req.openai_payload_fields is None:
        return True
    return field_name in req.openai_payload_fields


def should_include_gemini_field(req, field_name: str) -> bool:
    if req.gemini_payload_fields is None:
        return True
    return field_name in req.gemini_payload_fields


def decode_inline_data(data_str: str, context: str) -> bytes:
    normalized = (data_str or "").strip()
    if not normalized:
        return b""

    padding = (-len(normalized)) % 4
    if padding:
        normalized += "=" * padding

    try:
        return base64.b64decode(normalized, altchars=b"-_")
    except Exception as exc:
        logger.warning("[%s] Failed to decode inlineData: %s", context, exc)
        return b""


def parse_gemini_part(part: Dict[str, Any], content_idx: int, part_idx: int) -> UnifiedContentPart:
    """
    Parses a single Gemini Part dictionary into a UnifiedContentPart.
    """
    context = f"Gemini Content[{content_idx}] Part[{part_idx}]"
    known_part_fields = {
        "inlineData",
        "inline_data",
        "text",
        "thought",
        "thoughtSignature",
        "partMetadata",
        "videoMetadata",
        "functionCall",
        "functionResponse",
        "fileData",
        "executableCode",
        "codeExecutionResult",
    }
    warn_unknown_fields(part, known_part_fields, context)

    inline_data = part.get("inlineData") or part.get("inline_data")
    
    image_data_bytes = None
    mime_type = None

    if inline_data:
        data_str = inline_data.get("data", "") or ""
        mime_type = (
            inline_data.get("mimeType")
            or inline_data.get("mime_type")
            or "application/octet-stream"
        )

        if data_str:
            image_data_bytes = decode_inline_data(data_str, context)

        if image_data_bytes and (not mime_type or mime_type == "application/octet-stream"):
            detected = detect_content_type(image_data_bytes)
            if detected:
                mime_type = detected

    return UnifiedContentPart(
        text=part.get("text"),
        image_data=image_data_bytes,
        image_mime_type=mime_type,
        thought=part.get("thought"),
        thought_signature=part.get("thoughtSignature"),
        part_metadata=part.get("partMetadata"),
        video_metadata=part.get("videoMetadata"),
        function_call=part.get("functionCall"),
        function_response=part.get("functionResponse"),
        file_data=part.get("fileData"),
        executable_code=part.get("executableCode"),
        code_execution_result=part.get("codeExecutionResult"),
    )


def parse_gemini_contents(body: Dict[str, Any]) -> List[UnifiedMessage]:
    """
    Parses the 'contents' list from a Gemini request body into a list of UnifiedMessages.
    """
    parsed_messages: List[UnifiedMessage] = []

    contents = body.get("contents", []) or []
    for content_idx, content in enumerate(contents):
        warn_unknown_fields(content, {"role", "parts"}, f"Gemini Content[{content_idx}]")
        role = content.get("role") or "user"
        parts_raw = content.get("parts", []) or []
        parts: List[UnifiedContentPart] = []

        for part_idx, part in enumerate(parts_raw):
            parsed_part = parse_gemini_part(part, content_idx, part_idx)
            parts.append(parsed_part)

        parsed_messages.append(UnifiedMessage(role=role, parts=parts))

    return parsed_messages


def resolve_openai_size_for_gemini(size_value: Optional[str]) -> Optional[str]:
    if not size_value:
        return None

    cleaned = size_value.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered == "auto":
        return None

    token = lowered.split()[0]
    if token in OPENAI_ALLOWED_SIZES:
        return token

    logger.warning("[OpenAI Request] Unknown size '%s'. Omitting Gemini imageConfig.", cleaned)
    return None


def map_gemini_ar_to_openai_size(aspect_ratio: str) -> str:
    if aspect_ratio in {"3:2", "4:3", "16:9", "21:9"}:
        return "1536x1024"
    if aspect_ratio in {"2:3", "3:4", "9:16"}:
        return "1024x1536"
    return "1024x1024"


def derive_gemini_image_config_from_size(size: str) -> Tuple[str, str]:
    try:
        width_str, height_str = size.lower().split("x", 1)
        width = int(width_str)
        height = int(height_str)
    except Exception:
        return "1:1", "1K"

    if width == height:
        aspect_ratio = "1:1"
    elif width > height:
        ratio = width / max(1, height)
        if ratio >= 1.7:
            aspect_ratio = "16:9"
        elif ratio >= 1.3:
            aspect_ratio = "3:2"
        else:
            aspect_ratio = "4:3"
    else:
        ratio = height / max(1, width)
        if ratio >= 1.7:
            aspect_ratio = "9:16"
        elif ratio >= 1.3:
            aspect_ratio = "2:3"
        else:
            aspect_ratio = "3:4"

    max_side = max(width, height)
    image_size = "2K" if max_side >= 2048 else "1K"
    return aspect_ratio, image_size


def warn_openai_mapping_loss(req):
    """
    Warns if the UnifiedRequest contains data that cannot be represented in a simple OpenAI prompt.
    """
    unsupported_fields = []
    if req.generation_config:
        unsupported_fields.append("generation_config")
    if req.safety_settings:
        unsupported_fields.append("safety_settings")
    if req.tools:
        unsupported_fields.append("tools")
    if req.tool_config:
        unsupported_fields.append("tool_config")
    if req.system_instruction:
        unsupported_fields.append("system_instruction")
    if req.cached_content:
        unsupported_fields.append("cached_content")
    if unsupported_fields:
        logger.warning(
            "[OpenAI Payload] Dropping unsupported Gemini fields: %s",
            ", ".join(sorted(unsupported_fields)),
        )

    # Check for complex parts that will be lost or flattened
    has_complex = False
    for msg in req.messages:
        for part in msg.parts:
            if (
                part.function_call 
                or part.function_response 
                or part.file_data 
                or part.executable_code 
                or part.code_execution_result
                or part.video_metadata
            ):
                has_complex = True
                break
    
    if has_complex:
        logger.warning(
            "[OpenAI Payload] Messages contain structured parts (function calls, videos, etc.) "
            "that cannot be represented in OpenAI Images API and will be ignored or flattened."
        )
    elif len(req.messages) > 1:
        logger.warning(
            "[OpenAI Payload] Multiple message turns detected; they will be concatenated into a single prompt "
            "for the OpenAI API, which may change semantics."
        )


def warn_gemini_mapping_loss(req):
    openai_only_fields = []
    for field_name in (
        "style",
        "background",
        "moderation",
        "quality",
        "output_format",
        "output_compression",
        "partial_images",
        "stream",
        "input_fidelity",
        "user",
    ):
        value = getattr(req, field_name)
        if value not in (None, "", []):
            openai_only_fields.append(field_name)

    if openai_only_fields:
        logger.warning(
            "[Gemini Payload] Ignoring OpenAI-only fields with no Gemini equivalent: %s",
            ", ".join(sorted(openai_only_fields)),
        )