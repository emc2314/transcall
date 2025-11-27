import base64
from collections import defaultdict
from typing import Any, Dict, List, Set

from fastapi import HTTPException

from app.mappers import common
from app.schemas import UnifiedImageRequest, UnifiedImageResponse


def unified_to_gemini_payload(req: UnifiedImageRequest) -> Dict[str, Any]:
    common.warn_gemini_mapping_loss(req)

    size_for_gemini = common.resolve_openai_size_for_gemini(req.size)

    contents: List[Dict[str, Any]] = []
    
    if req.messages:
        for msg in req.messages:
            parts_payload: List[Dict[str, Any]] = []
            for part in msg.parts:
                part_payload: Dict[str, Any] = {}
                
                # Text
                if part.text is not None:
                    part_payload["text"] = part.text
                
                # Image
                if part.image_data:
                    data_b64 = base64.b64encode(part.image_data).decode("utf-8")
                    part_payload["inlineData"] = {
                        "mimeType": part.image_mime_type or "image/png",
                        "data": data_b64,
                    }
                
                # Advanced fields
                if part.thought is not None:
                    part_payload["thought"] = part.thought
                if part.thought_signature:
                    part_payload["thoughtSignature"] = part.thought_signature
                if part.part_metadata:
                    part_payload["partMetadata"] = part.part_metadata
                if part.video_metadata:
                    part_payload["videoMetadata"] = part.video_metadata
                if part.function_call:
                    part_payload["functionCall"] = part.function_call
                if part.function_response:
                    part_payload["functionResponse"] = part.function_response
                if part.file_data:
                    part_payload["fileData"] = part.file_data
                if part.executable_code:
                    part_payload["executableCode"] = part.executable_code
                if part.code_execution_result:
                    part_payload["codeExecutionResult"] = part.code_execution_result
                
                if part_payload:
                    parts_payload.append(part_payload)

            if parts_payload:
                content_payload: Dict[str, Any] = {"parts": parts_payload}
                content_payload["role"] = msg.role
                contents.append(content_payload)

    # Fallback for empty messages (should rarely happen if lift is correct)
    if not contents:
        # Try to use mask as a hint or just prompt
        if req.mask_image_bytes:
             raise HTTPException(
                status_code=400,
                detail="The 'mask' parameter is not supported for Gemini models via this endpoint.",
            )
        # Note: Prompt is computed from messages, so if messages are empty, prompt is empty.
        # But for safety/legacy
        pass 

    # Handle legacy Edit image inputs if they were not lifted into messages 
    # (But our lift logic DOES lift them, so this might be redundant unless direct object creation)
    # We will assume messages is the source of truth.

    payload: Dict[str, Any] = {"contents": contents}

    is_openai_source = req.gemini_payload_fields is None # Heuristic

    if common.should_include_gemini_field(req, "generationConfig"):
        generation_config_payload: Dict[str, Any] = {"candidateCount": req.n}
        if size_for_gemini:
            aspect_ratio, image_size = common.derive_gemini_image_config_from_size(size_for_gemini)
            generation_config_payload["imageConfig"] = {
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
            }

        if is_openai_source:
            generation_config_payload["responseModalities"] = ["TEXT", "IMAGE"]

        if req.generation_config:
            generation_config_payload.update(req.generation_config)

        payload["generationConfig"] = generation_config_payload

    if req.safety_settings:
        payload["safetySettings"] = req.safety_settings
    else:
        default_threshold = "OFF"
        payload["safetySettings"] = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": default_threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": default_threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": default_threshold},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": default_threshold},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": default_threshold},
        ]

    if req.tools and common.should_include_gemini_field(req, "tools"):
        payload["tools"] = req.tools

    if req.tool_config and common.should_include_gemini_field(req, "toolConfig"):
        payload["toolConfig"] = req.tool_config

    if req.system_instruction and common.should_include_gemini_field(req, "systemInstruction"):
        payload["systemInstruction"] = req.system_instruction

    if req.cached_content and common.should_include_gemini_field(req, "cachedContent"):
        payload["cachedContent"] = req.cached_content

    return payload


def unified_to_gemini_format(unified: UnifiedImageResponse) -> Dict[str, Any]:
    _warn_gemini_response_loss(unified)

    candidates_map: Dict[int, Dict[str, Any]] = {}
    parts_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    next_auto_index = 0
    existing_indices = [
        img.index for img in unified.images if img.index is not None
    ]
    if existing_indices:
        next_auto_index = max(existing_indices) + 1

    for img in unified.images:
        idx = img.index
        if idx is None:
            idx = next_auto_index
            next_auto_index += 1

        if idx not in candidates_map:
            candidate: Dict[str, Any] = {}
            if img.finish_reason:
                candidate["finishReason"] = img.finish_reason
            if img.safety_ratings:
                candidate["safetyRatings"] = img.safety_ratings
            if img.citation_metadata:
                candidate["citationMetadata"] = img.citation_metadata
            if img.grounding_metadata:
                candidate["groundingMetadata"] = img.grounding_metadata
            if img.token_count is not None:
                candidate["tokenCount"] = img.token_count
            candidate["index"] = idx

            if img.extra_info and "finishMessage" in img.extra_info:
                candidate["finishMessage"] = img.extra_info["finishMessage"]

            candidates_map[idx] = candidate

        part: Dict[str, Any] = {}

        if img.revised_prompt:
            part["text"] = img.revised_prompt

        if img.thought is not None:
            part["thought"] = img.thought

        if img.thought_signature:
            part["thoughtSignature"] = img.thought_signature

        if img.part_metadata:
            part["partMetadata"] = img.part_metadata

        if img.video_metadata:
            part["videoMetadata"] = img.video_metadata

        if img.function_call:
            part["functionCall"] = img.function_call
        if img.function_response:
            part["functionResponse"] = img.function_response
        if img.file_data:
            part["fileData"] = img.file_data
        if img.executable_code:
            part["executableCode"] = img.executable_code
        if img.code_execution_result:
            part["codeExecutionResult"] = img.code_execution_result

        if img.b64_json:
            part["inlineData"] = {"mimeType": img.mime_type, "data": img.b64_json}

        if part:
            parts_map[idx].append(part)

    final_candidates = []
    for idx in sorted(candidates_map.keys()):
        cand = candidates_map[idx]
        parts = parts_map[idx]
        if parts:
            cand["content"] = {"parts": parts}
        final_candidates.append(cand)

    resp: Dict[str, Any] = {"candidates": final_candidates}

    if unified.usage_source == "gemini" and unified.usage:
        resp["usageMetadata"] = unified.usage

    if unified.prompt_feedback:
        resp["promptFeedback"] = unified.prompt_feedback

    if unified.model_version:
        resp["modelVersion"] = unified.model_version

    return resp


def _warn_gemini_response_loss(unified: UnifiedImageResponse):
    if unified.usage_source == "gemini":
        return
    dropped_fields: Set[str] = set()
    for img in unified.images:
        if img.extra_info:
            dropped_fields.update(img.extra_info.keys())
    if unified.metadata:
        dropped_fields.update(unified.metadata.keys())
    if dropped_fields:
        common.logger.warning(
            "[Gemini Response] Ignoring OpenAI-specific fields while mapping to Gemini format: %s",
            ", ".join(sorted(dropped_fields)),
        )