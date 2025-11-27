import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.mappers import common
from app.schemas import (
    UnifiedImageRequest,
    UnifiedImageResponse,
    UnifiedImageResponseItem,
    UnifiedMessage,
    UnifiedContentPart,
    ProviderName,
)
from app.utils import detect_content_type


def openai_gen_to_unified(
    body: Dict, config: Dict, provider: ProviderName
) -> UnifiedImageRequest:
    known_fields = {
        "model",
        "prompt",
        "n",
        "size",
        "response_format",
        "user",
        "style",
        "background",
        "moderation",
        "quality",
        "output_format",
        "output_compression",
        "partial_images",
        "stream",
    }
    common.warn_unknown_fields(body, known_fields, "OpenAI Generation Request")

    provided_fields: Set[str] = set()
    for field in common.OPENAI_OPTIONAL_FIELDS:
        if body.get(field) is not None:
            provided_fields.add(field)

    prompt = body.get("prompt", "")
    messages = [
        UnifiedMessage(role="user", parts=[UnifiedContentPart(text=prompt)])
    ]

    return UnifiedImageRequest(
        target_model=body.get("model", "gpt-image-1"),
        provider=provider,
        messages=messages,
        n=body.get("n", 1),
        size=body.get("size"),
        response_format=body.get("response_format"),
        style=body.get("style"),
        background=body.get("background"),
        moderation=body.get("moderation"),
        quality=body.get("quality"),
        output_format=body.get("output_format"),
        output_compression=body.get("output_compression"),
        partial_images=body.get("partial_images"),
        stream=body.get("stream"),
        user=body.get("user"),
        openai_payload_fields=provided_fields,
    )


def openai_edit_to_unified(
    params: Dict,
    image_data_list: List[Tuple[bytes, str]],
    mask_data: Optional[Tuple[bytes, str]] = None,
    config: Optional[Dict] = None,
    provider: ProviderName = "openai",
    provided_fields: Optional[Set[str]] = None,
    image_field_names: Optional[List[str]] = None,
) -> UnifiedImageRequest:

    known_fields = {
        "model",
        "prompt",
        "n",
        "size",
        "response_format",
        "user",
        "background",
        "moderation",
        "quality",
        "output_format",
        "output_compression",
        "partial_images",
        "stream",
        "input_fidelity",
    }
    common.warn_unknown_fields(params, known_fields, "OpenAI Edit Request")

    target_model = params.get("model") or "gpt-image-1"

    # Construct Message with Image and Text parts
    parts: List[UnifiedContentPart] = []
    
    for img_bytes, raw_mime in image_data_list:
        mime = (
            detect_content_type(img_bytes) or raw_mime
            if (not raw_mime or raw_mime == "application/octet-stream")
            else raw_mime
        )
        parts.append(UnifiedContentPart(image_data=img_bytes, image_mime_type=mime))

    parts.append(UnifiedContentPart(text=params.get("prompt", "")))

    messages = [UnifiedMessage(role="user", parts=parts)]

    mask_bytes = mask_data[0] if mask_data else None
    mask_mime = None
    if mask_data:
        mask_mime = (
            detect_content_type(mask_data[0]) or mask_data[1]
            if (not mask_data[1] or mask_data[1] == "application/octet-stream")
            else mask_data[1]
        )

    return UnifiedImageRequest(
        target_model=target_model,
        provider=provider,
        messages=messages,
        n=params.get("n", 1),
        size=params.get("size"),
        response_format=params.get("response_format"),
        background=params.get("background"),
        moderation=params.get("moderation"),
        quality=params.get("quality"),
        output_format=params.get("output_format"),
        output_compression=params.get("output_compression"),
        partial_images=params.get("partial_images"),
        stream=params.get("stream"),
        user=params.get("user"),
        input_fidelity=params.get("input_fidelity"),
        mask_image_bytes=mask_bytes,
        mask_image_mime=mask_mime,
        openai_payload_fields=provided_fields,
    )


def openai_to_unified(resp: Dict) -> UnifiedImageResponse:
    images: List[UnifiedImageResponseItem] = []
    metadata = {k: v for k, v in resp.items() if k not in {"data", "created", "usage"}}
    
    mime_type = "image/png" 
    fmt = resp.get("output_format", "png").lower()
    if "jpeg" in fmt or "jpg" in fmt:
        mime_type = "image/jpeg"
    elif "webp" in fmt:
        mime_type = "image/webp"

    for item in resp.get("data", []):
        extra_info = {}
        known_item_fields = {"b64_json", "url", "revised_prompt"}
        for k, v in item.items():
            if k not in known_item_fields:
                extra_info[k] = v

        images.append(
            UnifiedImageResponseItem(
                b64_json=item.get("b64_json"),
                url=item.get("url"),
                mime_type=mime_type,
                revised_prompt=item.get("revised_prompt"),
                extra_info=extra_info if extra_info else None,
            )
        )

    return UnifiedImageResponse(
        images=images,
        created=resp.get("created", int(time.time())),
        usage=resp.get("usage"),
        usage_source="openai",
        metadata=metadata or None,
    )
