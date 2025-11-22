from typing import List, Optional, Literal, Any, Dict, Set
from pydantic import BaseModel

# --- Unified Internal Representation ---

class GeminiInlineData(BaseModel):
    mime_type: str = "application/octet-stream"
    data: bytes

class GeminiContentPart(BaseModel):
    text: Optional[str] = None
    inline_data: Optional[GeminiInlineData] = None
    thought: Optional[Any] = None
    thought_signature: Optional[str] = None
    part_metadata: Optional[Dict[str, Any]] = None
    video_metadata: Optional[Dict[str, Any]] = None
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Dict[str, Any]] = None
    file_data: Optional[Dict[str, Any]] = None
    executable_code: Optional[Dict[str, Any]] = None
    code_execution_result: Optional[Dict[str, Any]] = None

class GeminiContent(BaseModel):
    role: Optional[str] = None
    parts: List[GeminiContentPart]

class UnifiedImageRequest(BaseModel):
    """
    The canonical representation of an image generation/edit request 
    inside the system.
    """
    # Routing / Meta
    target_model: str  # The actual model name to use (e.g., "gpt-image-1")
    provider: str      # "openai" or "gemini"
    
    # Core Parameters (OpenAI & Gemini Intersection)
    prompt: str
    n: int = 1
    size: Optional[str] = None  # Preserve raw client preference (e.g., auto)
    response_format: Optional[Literal["url", "b64_json"]] = None

    # OpenAI Specifics
    style: Optional[Literal["vivid", "natural"]] = None # DALL-E 3
    background: Optional[Literal["transparent", "opaque", "auto"]] = None
    moderation: Optional[Literal["low", "auto"]] = None
    quality: Optional[Literal["high", "medium", "low", "standard", "hd"]] = None
    output_format: Optional[Literal["png", "jpeg", "webp"]] = None
    output_compression: Optional[int] = None # 0-100
    partial_images: Optional[int] = None # 0-3
    stream: Optional[bool] = None
    user: Optional[str] = None
    input_fidelity: Optional[Literal["high", "low"]] = None # For edits

    # Gemini Specifics (and Advanced Generation Params)
    # Captures temperature, topP, etc. 
    generation_config: Optional[Dict[str, Any]] = None 
    safety_settings: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_config: Optional[Dict[str, Any]] = None
    system_instruction: Optional[Dict[str, Any]] = None
    cached_content: Optional[str] = None
    gemini_contents: Optional[List[GeminiContent]] = None
    openai_payload_fields: Optional[Set[str]] = None
    gemini_payload_fields: Optional[Set[str]] = None

    # Input Data (Edits / Visual Prompting)
    input_image_bytes_list: Optional[List[bytes]] = None
    input_image_mime_list: Optional[List[str]] = None
    input_image_field_names: Optional[List[str]] = None
    mask_image_bytes: Optional[bytes] = None
    mask_image_mime: Optional[str] = None

class UnifiedImageResponseItem(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None # For when we support URL
    mime_type: str = "image/png"
    revised_prompt: Optional[str] = None
    finish_reason: Optional[str] = None
    safety_ratings: Optional[List[Dict[str, Any]]] = None
    
    # Detailed Gemini Metadata
    citation_metadata: Optional[Dict[str, Any]] = None
    grounding_metadata: Optional[Dict[str, Any]] = None
    token_count: Optional[int] = None
    index: Optional[int] = None
    
    # Gemini Part specific fields (if present in an image part)
    thought: Optional[Any] = None # Can be bool, text, or structured dict
    thought_signature: Optional[str] = None
    part_metadata: Optional[Dict[str, Any]] = None
    video_metadata: Optional[Dict[str, Any]] = None
    
    # Advanced Gemini Part Types
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Dict[str, Any]] = None
    file_data: Optional[Dict[str, Any]] = None
    executable_code: Optional[Dict[str, Any]] = None
    code_execution_result: Optional[Dict[str, Any]] = None
    
    # Catch-all for non-standard fields to avoid data loss without full raw dump
    extra_info: Optional[Dict[str, Any]] = None

class UnifiedImageResponse(BaseModel):
    """
    The canonical representation of the result.
    """
    images: List[UnifiedImageResponseItem]
    created: int
    usage: Optional[Dict[str, Any]] = None
    usage_source: Optional[Literal["openai", "gemini"]] = None
    
    # Request-level metadata (e.g. prompt feedback)
    prompt_feedback: Optional[Dict[str, Any]] = None
    model_version: Optional[str] = None
    
    # Flattened metadata from OpenAI (e.g. strict params reflected back)
    metadata: Optional[Dict[str, Any]] = None
