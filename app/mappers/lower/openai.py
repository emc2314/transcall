from collections import defaultdict
from typing import Any, Dict, List, Set

from app.mappers import common
from app.schemas import UnifiedImageRequest, UnifiedImageResponse


def unified_to_openai_payload(req: UnifiedImageRequest) -> Dict[str, Any]:
    common.warn_openai_mapping_loss(req)

    # Use the computed 'prompt' property which flattens all text messages
    prompt = req.prompt

    payload: Dict[str, Any] = {
        "model": req.target_model,
        "prompt": prompt,
    }

    # Determine size
    size = req.size
    if not size and req.generation_config:
        img_conf = req.generation_config.get("imageConfig")
        if img_conf and "aspectRatio" in img_conf:
            size = common.map_gemini_ar_to_openai_size(img_conf["aspectRatio"])

    optional_fields = {
        "n": req.n,
        "size": size,
        "response_format": req.response_format,
        "style": req.style,
        "background": req.background,
        "quality": req.quality,
        "output_format": req.output_format,
        "output_compression": req.output_compression,
        "partial_images": req.partial_images,
        "stream": req.stream,
        "user": req.user,
        "input_fidelity": req.input_fidelity,
    }

    if req.moderation is not None:
        optional_fields["moderation"] = req.moderation
    else:
        optional_fields["moderation"] = "low"

    for field, value in optional_fields.items():
        if value is None:
            continue
        if common.should_include_openai_field(req, field):
            payload[field] = value

    return payload


def unified_to_openai_format(unified: UnifiedImageResponse) -> Dict[str, Any]:
    # warn_openai_response_loss(unified) # Skipped for brevity/completeness

    grouped_items: Dict[int, Dict[str, Any]] = defaultdict(dict)
    first_valid_mime = None
    seen_indices: List[int] = []

    for img in unified.images:
        idx = img.index if img.index is not None else len(seen_indices)
        if idx not in grouped_items:
            seen_indices.append(idx)

        item = grouped_items[idx]

        if img.b64_json:
            item["b64_json"] = img.b64_json
            if not first_valid_mime and img.mime_type:
                first_valid_mime = img.mime_type

        if img.url:
            item["url"] = img.url

        if img.revised_prompt:
            existing_prompt = item.get("revised_prompt", "")
            if existing_prompt:
                item["revised_prompt"] = existing_prompt + "\n" + img.revised_prompt
            else:
                item["revised_prompt"] = img.revised_prompt

        if img.extra_info:
            item.update(img.extra_info)

    data_items: List[Dict[str, Any]] = []
    for idx in seen_indices:
        item = grouped_items[idx]
        if not item.get("b64_json") and not item.get("url"):
            continue
        data_items.append(item)

    resp: Dict[str, Any] = {"created": unified.created, "data": data_items}

    if unified.metadata:
        resp.update(unified.metadata)

    if "output_format" not in resp and first_valid_mime:
        if "png" in first_valid_mime:
            resp["output_format"] = "png"
        elif "jpeg" in first_valid_mime or "jpg" in first_valid_mime:
            resp["output_format"] = "jpeg"
        elif "webp" in first_valid_mime:
            resp["output_format"] = "webp"

    if unified.usage:
        if unified.usage_source == "gemini":
            u = unified.usage
            prompt_tokens = u.get("promptTokenCount", 0)
            resp["usage"] = {
                "input_tokens": prompt_tokens,
                "output_tokens": u.get("candidatesTokenCount", 0),
                "total_tokens": u.get("totalTokenCount", 0),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": u.get("candidatesTokenCount", 0),
                "input_tokens_details": {
                    "text_tokens": prompt_tokens,
                    "image_tokens": 0,
                },
            }
        elif unified.usage_source == "openai":
            resp["usage"] = unified.usage

    return resp