import base64
import time
import logging
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional, Set
from fastapi import HTTPException
from app.schemas import (
    UnifiedImageRequest,
    UnifiedImageResponse,
    UnifiedImageResponseItem,
    GeminiContent,
    GeminiContentPart,
    GeminiInlineData,
)

logger = logging.getLogger("Mappers")

class RequestMapper:
    _OPENAI_OPTIONAL_FIELDS: Set[str] = {
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
        "input_fidelity"
    }

    _GEMINI_OPTIONAL_FIELDS: Set[str] = {
        "generationConfig",
        "safetySettings",
        "tools",
        "toolConfig",
        "systemInstruction",
        "cachedContent"
    }
    _OPENAI_ALLOWED_SIZES: Set[str] = {"1024x1024", "1536x1024", "1024x1536"}
    _OPENAI_DEFAULT_SIZE: str = "1024x1024"

    @staticmethod
    def _gemini_contents_have_complex_parts(contents: List[GeminiContent]) -> bool:
        for content in contents:
            for part in content.parts:
                if part.inline_data or part.function_call or part.function_response or part.file_data or part.executable_code \
                        or part.code_execution_result or part.video_metadata or part.part_metadata \
                        or part.thought is not None or part.thought_signature:
                    return True
        return False

    @staticmethod
    def _gemini_contents_multi_turn(contents: List[GeminiContent]) -> bool:
        if len(contents) > 1:
            return True
        return any(len(content.parts) > 1 for content in contents)

    @staticmethod
    def _should_include_openai_field(req: UnifiedImageRequest, field_name: str) -> bool:
        if req.openai_payload_fields is None:
            return True
        return field_name in req.openai_payload_fields

    @staticmethod
    def _should_include_gemini_field(req: UnifiedImageRequest, field_name: str) -> bool:
        if req.gemini_payload_fields is None:
            return True
        return field_name in req.gemini_payload_fields

    @staticmethod
    def _resolve_openai_size_for_gemini(size_value: Optional[str]) -> Optional[str]:
        """
        Determine the concrete WxH string to use when translating an OpenAI request
        to Gemini (which can omit imageConfig when size isn't explicit).
        """
        if not size_value:
            return None

        cleaned = size_value.strip()
        if not cleaned:
            return None

        lowered = cleaned.lower()
        if lowered == "auto":
            return None

        token = lowered.split()[0]
        if token in RequestMapper._OPENAI_ALLOWED_SIZES:
            return token

        logger.warning(
            "[OpenAI Request] Unknown size '%s'. Omitting Gemini imageConfig.",
            cleaned
        )
        return None

    @staticmethod
    def _warn_openai_mapping_loss(req: UnifiedImageRequest):
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
                ", ".join(sorted(unsupported_fields))
            )

        if req.gemini_contents:
            if RequestMapper._gemini_contents_have_complex_parts(req.gemini_contents):
                logger.warning(
                    "[OpenAI Payload] Gemini structured parts (e.g., tool calls, thoughts, non-image attachments) "
                    "cannot be represented and will be flattened into a single prompt."
                )
            elif RequestMapper._gemini_contents_multi_turn(req.gemini_contents):
                logger.warning(
                    "[OpenAI Payload] Multiple Gemini turns/parts detected; they will be concatenated into one prompt "
                    "for the OpenAI API, which may change semantics."
                )

    @staticmethod
    def _warn_gemini_mapping_loss(req: UnifiedImageRequest):
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
                "[Gemini Payload] Ignoring OpenAI-only fields that have no Gemini equivalent: %s",
                ", ".join(sorted(openai_only_fields))
            )

    @staticmethod
    def _warn_unknown_fields(data: Dict[str, Any], known_fields: Set[str], context: str):
        unknown = set(data.keys()) - known_fields
        if unknown:
            logger.warning(f"[{context}] Unknown fields found and ignored: {unknown}")

    @staticmethod
    def _parse_gemini_part(part: Dict[str, Any], content_idx: int, part_idx: int) -> GeminiContentPart:
        context = f"Gemini Content[{content_idx}] Part[{part_idx}]"
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
            "codeExecutionResult"
        }
        RequestMapper._warn_unknown_fields(part, known_part_fields, context)

        inline_data_obj: Optional[GeminiInlineData] = None
        inline_data = part.get("inlineData")
        if inline_data:
            RequestMapper._warn_unknown_fields(inline_data, {"mimeType", "data"}, f"{context} inlineData")
            data_str = inline_data.get("data", "") or ""
            mime_type = inline_data.get("mimeType", "application/octet-stream")
            data_bytes = b""
            if data_str:
                try:
                    data_bytes = base64.b64decode(data_str)
                except Exception as exc:
                    logger.warning(f"[{context}] Failed to decode inlineData: {exc}")
            inline_data_obj = GeminiInlineData(mime_type=mime_type, data=data_bytes)

        return GeminiContentPart(
            text=part.get("text"),
            inline_data=inline_data_obj,
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

    @staticmethod
    def _parse_gemini_contents(body: Dict[str, Any]) -> Tuple[Optional[List[GeminiContent]], str, List[bytes], List[str]]:
        parsed_contents: List[GeminiContent] = []
        prompt_fragments: List[str] = []
        image_bytes: List[bytes] = []
        image_mimes: List[str] = []

        contents = body.get("contents", []) or []
        for content_idx, content in enumerate(contents):
            RequestMapper._warn_unknown_fields(content, {"role", "parts"}, f"Gemini Content[{content_idx}]")
            role = content.get("role")
            parts_raw = content.get("parts", []) or []
            parts: List[GeminiContentPart] = []

            for part_idx, part in enumerate(parts_raw):
                parsed_part = RequestMapper._parse_gemini_part(part, content_idx, part_idx)
                parts.append(parsed_part)

                if parsed_part.text:
                    prompt_fragments.append(parsed_part.text)

                if parsed_part.inline_data and parsed_part.inline_data.mime_type.lower().startswith("image/"):
                    image_bytes.append(parsed_part.inline_data.data)
                    image_mimes.append(parsed_part.inline_data.mime_type)

            parsed_contents.append(GeminiContent(role=role, parts=parts))

        prompt = " ".join(prompt_fragments).strip()
        return (parsed_contents or None, prompt, image_bytes, image_mimes)

    @staticmethod
    def _map_gemini_ar_to_openai_size(aspect_ratio: str) -> str:
        if aspect_ratio in {"3:2", "4:3", "16:9", "21:9"}:
            return "1536x1024"
        if aspect_ratio in {"2:3", "3:4", "9:16"}:
            return "1024x1536"
        return "1024x1024"

    @staticmethod
    def _derive_gemini_image_config_from_size(size: str) -> Tuple[str, str]:
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

    @staticmethod
    def openai_gen_to_unified(body: Dict, config: Dict, provider: str) -> UnifiedImageRequest:
        known_fields = {
            "model", "prompt", "n", "size", "response_format", "user",
            "style", "background", "moderation", "quality", 
            "output_format", "output_compression", "partial_images", "stream"
        }
        RequestMapper._warn_unknown_fields(body, known_fields, "OpenAI Generation Request")

        provided_fields: Set[str] = set()
        for field in RequestMapper._OPENAI_OPTIONAL_FIELDS:
            if body.get(field) is not None:
                provided_fields.add(field)

        return UnifiedImageRequest(
            target_model=body.get("model", "gpt-image-1"),
            provider=provider,
            prompt=body.get("prompt", ""),
            n=body.get("n", 1),
            size=body.get("size"),
            response_format=body.get("response_format"),
            
            # Extended OpenAI parameters
            style=body.get("style"),
            background=body.get("background"),
            moderation=body.get("moderation"),
            quality=body.get("quality"),
            output_format=body.get("output_format"),
            output_compression=body.get("output_compression"),
            partial_images=body.get("partial_images"),
            stream=body.get("stream"),
            user=body.get("user"),
            openai_payload_fields=provided_fields
        )

    @staticmethod
    def openai_edit_to_unified(
        params: Dict, 
        image_data_list: List[Tuple[bytes, str]],
        mask_data: Optional[Tuple[bytes, str]] = None,
        config: Optional[Dict] = None,
        provider: str = "openai",
        provided_fields: Optional[Set[str]] = None,
        image_field_names: Optional[List[str]] = None,
    ) -> UnifiedImageRequest:
        
        known_fields = {
            "model", "prompt", "n", "size", "response_format", "user",
            "background", "moderation", "quality", "output_format", 
            "output_compression", "partial_images", "stream", "input_fidelity"
        }
        RequestMapper._warn_unknown_fields(params, known_fields, "OpenAI Edit Request")

        target_model = params.get("model") or "gpt-image-1"

        return UnifiedImageRequest(
            target_model=target_model,
            provider=provider,
            prompt=params.get("prompt", ""),
            n=params.get("n", 1),
            size=params.get("size"),
            response_format=params.get("response_format"),

            # Extended OpenAI parameters
            background=params.get("background"),
            moderation=params.get("moderation"),
            quality=params.get("quality"),
            output_format=params.get("output_format"),
            output_compression=params.get("output_compression"),
            partial_images=params.get("partial_images"),
            stream=params.get("stream"),
            user=params.get("user"),
            input_fidelity=params.get("input_fidelity"),

            # Image data for edits
            input_image_bytes_list=[img[0] for img in image_data_list],
            input_image_mime_list=[img[1] for img in image_data_list],
            input_image_field_names=image_field_names,
            mask_image_bytes=mask_data[0] if mask_data else None,
            mask_image_mime=mask_data[1] if mask_data else None,
            openai_payload_fields=provided_fields
        )

    @staticmethod
    def gemini_content_to_unified(body: Dict, model_name: str, config: Dict, provider: str) -> UnifiedImageRequest:
        known_fields = {
            "contents", "tools", "toolConfig", "safetySettings", 
            "systemInstruction", "generationConfig", "cachedContent"
        }
        RequestMapper._warn_unknown_fields(body, known_fields, "Gemini Content Request")

        parsed_contents, prompt, img_bytes, img_mimes = RequestMapper._parse_gemini_contents(body)

        provided_gemini_fields: Set[str] = set()
        for field in RequestMapper._GEMINI_OPTIONAL_FIELDS:
            if body.get(field) is not None:
                provided_gemini_fields.add(field)
        
        gen_config = body.get("generationConfig", {}) or {}
        n = gen_config.get("candidateCount", 1)
        image_config = gen_config.get("imageConfig", {})
        aspect_ratio = image_config.get("aspectRatio") or "1:1"

        # Extract other generic generation config
        known_gen_config_fields = {"candidateCount", "imageConfig", "stopSequences", "maxOutputTokens", 
                                   "temperature", "topP", "topK", "seed", "responseMimeType", "responseSchema",
                                   "responseLogprobs", "logprobs"}
        
        other_gen_config = {k: v for k, v in gen_config.items() if k not in {"candidateCount", "imageConfig"}}
        RequestMapper._warn_unknown_fields(gen_config, known_gen_config_fields, "Gemini Generation Config")

        if provider == "gemini":
            normalized_size = RequestMapper._map_gemini_ar_to_openai_size(aspect_ratio)
            target_model = model_name
        else:
            # If mapped to OpenAI
            normalized_size = RequestMapper._map_gemini_ar_to_openai_size(aspect_ratio)
            target_model = "gpt-image-1"

        return UnifiedImageRequest(
            target_model=target_model,
            provider=provider,
            prompt=prompt,
            n=n,
            size=normalized_size,
            response_format="b64_json",
            
            # Gemini Specifics
            generation_config=other_gen_config if other_gen_config else None,
            safety_settings=body.get("safetySettings"),
            tools=body.get("tools"),
            tool_config=body.get("toolConfig"),
            system_instruction=body.get("systemInstruction"),
            cached_content=body.get("cachedContent"),
            gemini_contents=parsed_contents,
            gemini_payload_fields=provided_gemini_fields,
            
            # Visual Prompting
            input_image_bytes_list=img_bytes if img_bytes else None,
            input_image_mime_list=img_mimes if img_mimes else None
        )

    @staticmethod
    def unified_to_openai_payload(req: UnifiedImageRequest) -> Dict[str, Any]:
        RequestMapper._warn_openai_mapping_loss(req)

        payload: Dict[str, Any] = {
            "model": req.target_model,
            "prompt": req.prompt,
        }

        optional_fields = {
            "n": req.n,
            "size": req.size,
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

        # moderation default for OpenAI payloads if not provided
        if req.moderation is not None:
            optional_fields["moderation"] = req.moderation
        else:
            optional_fields["moderation"] = "low"

        for field, value in optional_fields.items():
            if value is None:
                continue
            if RequestMapper._should_include_openai_field(req, field):
                payload[field] = value

        return payload

    @staticmethod
    def unified_to_gemini_payload(req: UnifiedImageRequest) -> Dict[str, Any]:
        RequestMapper._warn_gemini_mapping_loss(req)

        size_for_gemini = RequestMapper._resolve_openai_size_for_gemini(req.size)

        # 1. Construct Contents
        if req.gemini_contents:
            contents: List[Dict[str, Any]] = []
            for content in req.gemini_contents:
                parts_payload: List[Dict[str, Any]] = []
                for part in content.parts:
                    part_payload: Dict[str, Any] = {}
                    if part.text is not None:
                        part_payload["text"] = part.text
                    if part.inline_data:
                        data_b64 = base64.b64encode(part.inline_data.data).decode("utf-8") if part.inline_data.data else ""
                        part_payload["inlineData"] = {
                            "mimeType": part.inline_data.mime_type,
                            "data": data_b64
                        }
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

                content_payload: Dict[str, Any] = {"parts": parts_payload}
                if content.role:
                    content_payload["role"] = content.role
                contents.append(content_payload)

            if not contents:
                contents = [{"parts": [{"text": req.prompt}]}]

        else:
            # Fallback to reconstructing from prompt/images (e.g. from OpenAI request)
            parts: List[Dict[str, Any]] = [{"text": req.prompt}]
            
            # Handle Edits (Visual Prompting)
            if req.input_image_bytes_list:
                if req.mask_image_bytes:
                     raise HTTPException(
                        status_code=400, 
                        detail="The 'mask' parameter is not supported for Gemini models via this endpoint."
                    )

                mime_list = req.input_image_mime_list or []
                for idx, img_bytes in enumerate(req.input_image_bytes_list):
                    b64_img = base64.b64encode(img_bytes).decode("utf-8")
                    mime_type = mime_list[idx] if idx < len(mime_list) else "image/png"
                    parts.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": b64_img
                        }
                    })
            contents = [{"parts": parts}]

        # 2. Construct Payload
        payload: Dict[str, Any] = {
            "contents": contents
        }

        if RequestMapper._should_include_gemini_field(req, "generationConfig"):
            generation_config_payload: Dict[str, Any] = {
                "candidateCount": req.n
            }
            if size_for_gemini:
                aspect_ratio, image_size = RequestMapper._derive_gemini_image_config_from_size(size_for_gemini)
                generation_config_payload["imageConfig"] = {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size
                }
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
            
        if req.tools and RequestMapper._should_include_gemini_field(req, "tools"):
            payload["tools"] = req.tools
            
        if req.tool_config and RequestMapper._should_include_gemini_field(req, "toolConfig"):
            payload["toolConfig"] = req.tool_config
            
        if req.system_instruction and RequestMapper._should_include_gemini_field(req, "systemInstruction"):
            payload["systemInstruction"] = req.system_instruction
            
        if req.cached_content and RequestMapper._should_include_gemini_field(req, "cachedContent"):
            payload["cachedContent"] = req.cached_content

        return payload


class ResponseMapper:

    @staticmethod
    def _mime_from_openai_format(output_format: Optional[str]) -> str:
        fmt = (output_format or "png").lower()
        return {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "webp": "image/webp"
        }.get(fmt, "image/png")

    @staticmethod
    def _collect_openai_metadata(resp: Dict[str, Any]) -> Dict[str, Any]:
        # Collect EVERYTHING except the standard data/created/usage fields
        excluded = {"data", "created", "usage"}
        return {k: v for k, v in resp.items() if k not in excluded}

    @staticmethod
    def _warn_openai_response_loss(unified: UnifiedImageResponse):
        if unified.usage_source == "openai":
            return
        dropped_fields: Set[str] = set()
        for img in unified.images:
            if img.finish_reason:
                dropped_fields.add("finish_reason")
            if img.safety_ratings:
                dropped_fields.add("safety_ratings")
            if img.citation_metadata:
                dropped_fields.add("citation_metadata")
            if img.grounding_metadata:
                dropped_fields.add("grounding_metadata")
            if img.token_count is not None:
                dropped_fields.add("token_count")
            if img.thought is not None:
                dropped_fields.add("thought")
            if img.thought_signature:
                dropped_fields.add("thought_signature")
            if img.part_metadata:
                dropped_fields.add("part_metadata")
            if img.video_metadata:
                dropped_fields.add("video_metadata")
            if img.function_call:
                dropped_fields.add("function_call")
            if img.function_response:
                dropped_fields.add("function_response")
            if img.file_data:
                dropped_fields.add("file_data")
            if img.executable_code:
                dropped_fields.add("executable_code")
            if img.code_execution_result:
                dropped_fields.add("code_execution_result")
        if dropped_fields:
            logger.warning(
                "[OpenAI Response] Dropping Gemini-specific fields that have no OpenAI equivalent: %s",
                ", ".join(sorted(dropped_fields))
            )

    @staticmethod
    def _warn_gemini_response_loss(unified: UnifiedImageResponse):
        if unified.usage_source == "gemini":
            return
        dropped_fields: Set[str] = set()
        for img in unified.images:
            if img.extra_info:
                dropped_fields.add("extra_info")
        if unified.metadata:
            dropped_fields.update(unified.metadata.keys())
        if dropped_fields:
            logger.warning(
                "[Gemini Response] Ignoring OpenAI-specific fields while mapping to Gemini format: %s",
                ", ".join(sorted(dropped_fields))
            )

    @staticmethod
    def openai_to_unified(resp: Dict) -> UnifiedImageResponse:
        # known_fields check is less relevant now that we capture everything else as metadata, 
        # but we can still keep it for sanity or remove strict warning.
        # Let's rely on capturing everything into metadata for forward compatibility.

        images: List[UnifiedImageResponseItem] = []
        metadata = ResponseMapper._collect_openai_metadata(resp)
        mime_type = ResponseMapper._mime_from_openai_format(resp.get("output_format"))
        
        for item in resp.get("data", []):
            extra_info = {}
            known_item_fields = {"b64_json", "url", "revised_prompt"}
            # Capture anything extra that isn't strictly standard but might be returned
            for k, v in item.items():
                if k not in known_item_fields:
                    extra_info[k] = v

            images.append(UnifiedImageResponseItem(
                b64_json=item.get("b64_json"),
                url=item.get("url"),
                mime_type=mime_type,
                revised_prompt=item.get("revised_prompt"),
                extra_info=extra_info if extra_info else None
            ))
        
        return UnifiedImageResponse(
            images=images,
            created=resp.get("created", int(time.time())),
            usage=resp.get("usage"),
            usage_source="openai",
            metadata=metadata or None
        )

    @staticmethod
    def gemini_to_unified(resp: Dict) -> UnifiedImageResponse:
        known_fields = {"candidates", "promptFeedback", "usageMetadata", "modelVersion"}
        RequestMapper._warn_unknown_fields(resp, known_fields, "Gemini Response")

        images: List[UnifiedImageResponseItem] = []
        candidates = resp.get("candidates", [])
        for cand in candidates:
            known_cand_fields = {"content", "finishReason", "safetyRatings", "citationMetadata", 
                                 "tokenCount", "groundingAttributions", "groundingMetadata", 
                                 "avgLogprobs", "logprobsResult", "index", "finishMessage"}
            RequestMapper._warn_unknown_fields(cand, known_cand_fields, "Gemini Candidate")

            finish_reason = cand.get("finishReason")
            safety_ratings = cand.get("safetyRatings")
            
            # Extract specific Gemini candidate-level metadata
            citation_metadata = cand.get("citationMetadata")
            grounding_metadata = cand.get("groundingMetadata")
            token_count = cand.get("tokenCount")
            index = cand.get("index")
            finish_message = cand.get("finishMessage")

            parts = cand.get("content", {}).get("parts", [])
            
            for p in parts:
                # For each part, create a separate UnifiedImageResponseItem
                
                image_data = None
                mime_type = "image/png"
                text_content = None
                thought_val = None
                thought_sig = None
                part_meta = None
                video_meta = None
                
                # Advanced fields
                func_call = None
                func_resp = None
                file_data = None
                exe_code = None
                code_res = None
                
                # Check Part Type
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
                
                # Warn for other unknown part fields
                known_part_fields = {"inlineData", "text", "thought", "thoughtSignature", "partMetadata", "videoMetadata", 
                                     "functionCall", "functionResponse", "fileData", "executableCode", "codeExecutionResult"}
                RequestMapper._warn_unknown_fields(p, known_part_fields, "Gemini Content Part")

                # Create Item
                images.append(UnifiedImageResponseItem(
                    b64_json=image_data, # Optional
                    mime_type=mime_type,
                    revised_prompt=text_content, # Using revised_prompt as "Text Content" carrier
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
                    extra_info={"finishMessage": finish_message} if finish_message else None
                ))

        return UnifiedImageResponse(
            images=images,
            created=int(time.time()),
            usage=resp.get("usageMetadata"),
            usage_source="gemini",
            prompt_feedback=resp.get("promptFeedback"),
            model_version=resp.get("modelVersion")
        )

    @staticmethod
    def unified_to_openai_format(unified: UnifiedImageResponse) -> Dict[str, Any]:
        ResponseMapper._warn_openai_response_loss(unified)

        data_items: List[Dict[str, Any]] = []
        for img in unified.images:
            item: Dict[str, Any] = {}
            if img.b64_json:
                item["b64_json"] = img.b64_json
            if img.url:
                item["url"] = img.url
            if img.revised_prompt:
                item["revised_prompt"] = img.revised_prompt
            
            # Re-attach extra info that might have come from OpenAI original response
            if img.extra_info:
                item.update(img.extra_info)
                
            data_items.append(item)
        
        resp: Dict[str, Any] = {
            "created": unified.created,
            "data": data_items
        }

        if unified.metadata:
            resp.update(unified.metadata)
        
        if unified.usage_source == "openai" and unified.usage:
            resp["usage"] = unified.usage
        
        return resp

    @staticmethod
    def unified_to_gemini_format(unified: UnifiedImageResponse) -> Dict[str, Any]:
        ResponseMapper._warn_gemini_response_loss(unified)

        # Group items by index to reconstruct Candidates
        candidates_map: Dict[int, Dict[str, Any]] = {}
        parts_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        
        next_auto_index = 0
        existing_indices = [img.index for img in unified.images if img.index is not None]
        if existing_indices:
            next_auto_index = max(existing_indices) + 1

        for img in unified.images:
            idx = img.index
            if idx is None:
                idx = next_auto_index
                next_auto_index += 1
            
            # Initialize Candidate Structure if new index
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
                # Explicitly set index in candidate
                candidate["index"] = idx
                
                if img.extra_info and "finishMessage" in img.extra_info:
                    candidate["finishMessage"] = img.extra_info["finishMessage"]
                
                candidates_map[idx] = candidate

            # Construct Part
            # Text
            if img.revised_prompt:
                parts_map[idx].append({"text": img.revised_prompt})

            # Thought (can be Any, but typically matches Gemini part structure)
            if img.thought is not None:
                parts_map[idx].append({"thought": img.thought})
            
            # Thought Signature
            if img.thought_signature:
                parts_map[idx].append({"thoughtSignature": img.thought_signature})
            
            # Part Metadata
            if img.part_metadata:
                parts_map[idx].append({"partMetadata": img.part_metadata})
            
            # Video Metadata
            if img.video_metadata:
                parts_map[idx].append({"videoMetadata": img.video_metadata})
                
            # Advanced Fields
            if img.function_call:
                parts_map[idx].append({"functionCall": img.function_call})
            if img.function_response:
                parts_map[idx].append({"functionResponse": img.function_response})
            if img.file_data:
                parts_map[idx].append({"fileData": img.file_data})
            if img.executable_code:
                parts_map[idx].append({"executableCode": img.executable_code})
            if img.code_execution_result:
                parts_map[idx].append({"codeExecutionResult": img.code_execution_result})

            # Image
            if img.b64_json:
                parts_map[idx].append({
                    "inlineData": {
                        "mimeType": img.mime_type,
                        "data": img.b64_json
                    }
                })
        
        # Assemble final list
        final_candidates = []
        sorted_indices = sorted(candidates_map.keys())
        
        for idx in sorted_indices:
            cand = candidates_map[idx]
            parts = parts_map[idx]
            if parts:
                cand["content"] = {"parts": parts}
            final_candidates.append(cand)
            
        resp: Dict[str, Any] = {
            "candidates": final_candidates
        }
        
        if unified.usage_source == "gemini" and unified.usage:
            resp["usageMetadata"] = unified.usage
            
        if unified.prompt_feedback:
            resp["promptFeedback"] = unified.prompt_feedback
            
        if unified.model_version:
            resp["modelVersion"] = unified.model_version
        
        return resp
