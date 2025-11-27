from typing import Any, Dict, List, Optional, Set

from app.mappers import common
from app.schemas import (
    UnifiedImageRequest,
    UnifiedImageResponse,
    UnifiedImageResponseItem,
    ProviderName,
)


def gemini_content_to_unified(
    body: Dict, model_name: str, config: Dict, provider: ProviderName
) -> UnifiedImageRequest:
    known_fields = {
        "contents",
        "tools",
        "toolConfig",
        "safetySettings",
        "systemInstruction",
        "generationConfig",
        "cachedContent",
    }
    common.warn_unknown_fields(body, known_fields, "Gemini Content Request")

    parsed_messages = common.parse_gemini_contents(body)

    provided_fields: Set[str] = set()
    for field in common.GEMINI_OPTIONAL_FIELDS:
        if body.get(field) is not None:
            provided_fields.add(field)

    gen_config = body.get("generationConfig", {}) or {}
    n = gen_config.get("candidateCount", 1)

    other_gen_config = {k: v for k, v in gen_config.items() if k not in {"candidateCount"}}
    common.warn_unknown_fields(
        gen_config,
        {
            "candidateCount",
            "imageConfig",
            "stopSequences",
            "maxOutputTokens",
            "temperature",
            "topP",
            "topK",
            "seed",
            "responseMimeType",
            "responseSchema",
            "responseLogprobs",
            "logprobs",
            "responseModalities",
        },
        "Gemini Generation Config",
    )

    return UnifiedImageRequest(
        target_model=model_name,
        provider=provider,
        messages=parsed_messages,
        n=n,
        size=None,
        response_format=None,
        generation_config=other_gen_config if other_gen_config else None,
        safety_settings=body.get("safetySettings"),
        tools=body.get("tools"),
        tool_config=body.get("toolConfig"),
        system_instruction=body.get("systemInstruction"),
        cached_content=body.get("cachedContent"),
        gemini_payload_fields=provided_fields,
    )


def gemini_to_unified(resp: Dict[str, Any]) -> UnifiedImageResponse:
    common.warn_unknown_fields(resp, {"candidates", "promptFeedback", "usageMetadata", "modelVersion"}, "Gemini Response")

    images: List[UnifiedImageResponseItem] = []
    candidates = resp.get("candidates", [])
    for cand_idx, cand in enumerate(candidates):
        known_cand_fields = {
            "content",
            "finishReason",
            "safetyRatings",
            "citationMetadata",
            "tokenCount",
            "groundingAttributions",
            "groundingMetadata",
            "avgLogprobs",
            "logprobsResult",
            "index",
            "finishMessage",
        }
        common.warn_unknown_fields(cand, known_cand_fields, "Gemini Candidate")

        finish_reason = cand.get("finishReason")
        safety_ratings = cand.get("safetyRatings")
        citation_metadata = cand.get("citationMetadata")
        grounding_metadata = cand.get("groundingMetadata")
        token_count = cand.get("tokenCount")
        index = cand.get("index")
        if index is None:
            index = cand_idx

        finish_message = cand.get("finishMessage")

        parts = cand.get("content", {}).get("parts", [])

        for p in parts:
            image_data = None
            mime_type = "image/png"
            text_content = None
            thought_val = None
            thought_sig = None
            part_meta = None
            video_meta = None
            func_call = None
            func_resp = None
            file_data = None
            exe_code = None
            code_res = None

            if "inlineData" in p:
                image_data = p["inlineData"]["data"]
                mime_type = p["inlineData"].get("mimeType", "image/png")

            if "text" in p:
                text_content = p["text"]

            if "thought" in p:
                thought_val = p["thought"]

            if "thoughtSignature" in p:
                thought_sig = p["thoughtSignature"]

            if "partMetadata" in p:
                part_meta = p["partMetadata"]

            if "videoMetadata" in p:
                video_meta = p["videoMetadata"]

            if "functionCall" in p:
                func_call = p["functionCall"]

            if "functionResponse" in p:
                func_resp = p["functionResponse"]

            if "fileData" in p:
                file_data = p["fileData"]

            if "executableCode" in p:
                exe_code = p["executableCode"]

            if "codeExecutionResult" in p:
                code_res = p["codeExecutionResult"]

            known_part_fields = {
                "inlineData",
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
            common.warn_unknown_fields(p, known_part_fields, "Gemini Content Part")

            images.append(
                UnifiedImageResponseItem(
                    b64_json=image_data,
                    mime_type=mime_type,
                    revised_prompt=text_content,
                    finish_reason=finish_reason,
                    safety_ratings=safety_ratings,
                    citation_metadata=citation_metadata,
                    grounding_metadata=grounding_metadata,
                    token_count=token_count,
                    index=index,
                    thought=thought_val,
                    thought_signature=thought_sig,
                    part_metadata=part_meta,
                    video_metadata=video_meta,
                    function_call=func_call,
                    function_response=func_resp,
                    file_data=file_data,
                    executable_code=exe_code,
                    code_execution_result=code_res,
                    extra_info={"finishMessage": finish_message} if finish_message else None,
                )
            )

    return UnifiedImageResponse(
        images=images,
        created=resp.get("created", 0), 
        usage=resp.get("usageMetadata"),
        usage_source="gemini",
        prompt_feedback=resp.get("promptFeedback"),
        model_version=resp.get("modelVersion"),
    )
