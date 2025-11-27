from app.mappers.lift.openai import (
    openai_edit_to_unified,
    openai_gen_to_unified,
    openai_to_unified,
)
from app.mappers.lift.gemini import gemini_content_to_unified, gemini_to_unified
from app.mappers.lower.openai import unified_to_openai_format, unified_to_openai_payload
from app.mappers.lower.gemini import unified_to_gemini_format, unified_to_gemini_payload


class RequestMapper:
    openai_gen_to_unified = staticmethod(openai_gen_to_unified)
    openai_edit_to_unified = staticmethod(openai_edit_to_unified)
    gemini_content_to_unified = staticmethod(gemini_content_to_unified)
    unified_to_openai_payload = staticmethod(unified_to_openai_payload)
    unified_to_gemini_payload = staticmethod(unified_to_gemini_payload)


class ResponseMapper:
    openai_to_unified = staticmethod(openai_to_unified)
    gemini_to_unified = staticmethod(gemini_to_unified)
    unified_to_openai_format = staticmethod(unified_to_openai_format)
    unified_to_gemini_format = staticmethod(unified_to_gemini_format)


__all__ = ["RequestMapper", "ResponseMapper"]
