import pytest
import respx
from httpx import Response

from app.schemas import UnifiedImageRequest
from app.providers.openai import OpenAIProvider
from app.providers.gemini import GeminiProvider


@pytest.mark.asyncio
@respx.mock
async def test_openai_call_uses_bearer_header(
    openai_config, openai_generation_response
):
    req = UnifiedImageRequest(
        target_model="gpt-image-1",
        provider="openai",
        prompt="A test image",
        n=1,
        response_format="b64_json",
    )

    route = respx.post("https://api.openai.com/v1/images/generations").mock(
        return_value=Response(200, json=openai_generation_response)
    )

    provider = OpenAIProvider(openai_config)
    await provider.generate(req)

    assert route.called
    upstream_request = route.calls.last.request
    assert upstream_request.headers["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
@respx.mock
async def test_gemini_call_attaches_api_key(
    gemini_config, gemini_generate_content_response
):
    req = UnifiedImageRequest(
        target_model="gemini-3-pro-image-preview",
        provider="gemini",
        prompt="Describe a nebula",
        n=1,
    )

    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
    ).mock(return_value=Response(200, json=gemini_generate_content_response))

    provider = GeminiProvider(gemini_config)
    await provider.generate(req)

    assert route.called
    upstream_request = route.calls.last.request
    assert upstream_request.url.params["key"] == "gemini-key"
