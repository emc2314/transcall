from typing import List, Optional, Literal, Any, Dict, Set
from pydantic import BaseModel, Field, computed_field

# --- Unified Internal Representation ---

ProviderName = Literal["openai", "gemini", "vertexai"]


class UnifiedContentPart(BaseModel):
    """
    A neutral representation of a part of a message.
    Can represent text, images, tool calls, etc.
    """
    text: Optional[str] = None
    
    # Image Data (Raw bytes, decoded from base64 or read from file)
    image_data: Optional[bytes] = None
    image_mime_type: Optional[str] = "image/png"
    
    # Advanced Interaction Fields (Tooling, Thoughts, etc.)
    thought: Optional[Any] = None
    thought_signature: Optional[str] = None
    
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Dict[str, Any]] = None
    
    file_data: Optional[Dict[str, Any]] = None 
    video_metadata: Optional[Dict[str, Any]] = None
    
    executable_code: Optional[Dict[str, Any]] = None
    code_execution_result: Optional[Dict[str, Any]] = None
    
    part_metadata: Optional[Dict[str, Any]] = None


class UnifiedMessage(BaseModel):
    """
    A single message in a conversation history.
    """
    role: str = "user" 
    parts: List[UnifiedContentPart] = Field(default_factory=list)


class UnifiedImageRequest(BaseModel):
    """
    The canonical representation of an image generation/edit request
    inside the system.
    """

    # Routing / Meta
    target_model: str
    provider: ProviderName

    # Core Content
    messages: List[UnifiedMessage] = Field(default_factory=list)

    # Generation Parameters
    n: int = 1
    size: Optional[str] = None 
    response_format: Optional[Literal["url", "b64_json"]] = None

    # OpenAI Specifics
    style: Optional[Literal["vivid", "natural"]] = None
    background: Optional[Literal["transparent", "opaque", "auto"]] = None
    moderation: Optional[Literal["low", "auto"]] = None
    quality: Optional[Literal["high", "medium", "low", "standard", "hd"]] = None
    output_format: Optional[Literal["png", "jpeg", "webp"]] = None
    output_compression: Optional[int] = None
    partial_images: Optional[int] = None
    stream: Optional[bool] = None
    user: Optional[str] = None
    input_fidelity: Optional[Literal["high", "low"]] = None

    # Gemini Specifics
    generation_config: Optional[Dict[str, Any]] = None
    safety_settings: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_config: Optional[Dict[str, Any]] = None
    system_instruction: Optional[Dict[str, Any]] = None
    cached_content: Optional[str] = None
    
    openai_payload_fields: Optional[Set[str]] = None
    gemini_payload_fields: Optional[Set[str]] = None

    mask_image_bytes: Optional[bytes] = None
    mask_image_mime: Optional[str] = None

    @computed_field
    def prompt(self) -> str:
        """
        Compatibility property: Flattens all text parts from all messages
        into a single prompt string.
        """
        fragments = []
        for msg in self.messages:
            for part in msg.parts:
                if part.text:
                    fragments.append(part.text)
        return " ".join(fragments).strip()


class UnifiedImageResponseItem(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    mime_type: str = "image/png"
    revised_prompt: Optional[str] = None
    finish_reason: Optional[str] = None
    safety_ratings: Optional[List[Dict[str, Any]]] = None

    # Detailed Metadata
    citation_metadata: Optional[Dict[str, Any]] = None
    grounding_metadata: Optional[Dict[str, Any]] = None
    token_count: Optional[int] = None
    index: Optional[int] = None

    # Part specific fields
    thought: Optional[Any] = None
    thought_signature: Optional[str] = None
    part_metadata: Optional[Dict[str, Any]] = None
    video_metadata: Optional[Dict[str, Any]] = None

    # Advanced Part Types
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Dict[str, Any]] = None
    file_data: Optional[Dict[str, Any]] = None
    executable_code: Optional[Dict[str, Any]] = None
    code_execution_result: Optional[Dict[str, Any]] = None

    # Catch-all
    extra_info: Optional[Dict[str, Any]] = None


class UnifiedImageResponse(BaseModel):
    """
    The canonical representation of the result.
    """
    images: List[UnifiedImageResponseItem]
    created: int
    usage: Optional[Dict[str, Any]] = None
    usage_source: Optional[Literal["openai", "gemini"]] = None
    prompt_feedback: Optional[Dict[str, Any]] = None
    model_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
