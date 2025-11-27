import pytest
from app.mappers import RequestMapper
from app.schemas import UnifiedImageRequest


def test_gemini_content_lift_snapshot(
    gemini_generate_content_request, gemini_config, snapshot
):
    unified = RequestMapper.gemini_content_to_unified(
        gemini_generate_content_request,
        model_name="gemini-3-pro-image-preview",
        config=gemini_config,
        provider="gemini",
    )
    assert unified.model_dump() == snapshot


@pytest.mark.parametrize("size", [None, "auto", "1024x1024"])
def test_openai_to_gemini_size_lowering(size):
    req = UnifiedImageRequest(
        target_model="gemini-3-pro-image-preview",
        provider="gemini",
        prompt="A beach at dusk",
        n=2,
        size=size,
    )
    payload = RequestMapper.unified_to_gemini_payload(req)
    gen_conf = payload["generationConfig"]
    assert gen_conf["candidateCount"] == 2
    if size in (None, "auto"):
        assert (
            "imageConfig" not in gen_conf
            or gen_conf["imageConfig"]["aspectRatio"] == "1:1"
        )
    else:
        assert gen_conf["imageConfig"]["aspectRatio"] == "1:1"


def test_gemini_lowering_snapshot():
    req = UnifiedImageRequest(
        target_model="gemini-3-pro-image-preview",
        provider="gemini",
        prompt="A mountain reflected on a lake",
        n=1,
        size="1024x1024",
    )
    payload = RequestMapper.unified_to_gemini_payload(req)
    assert "contents" in payload
    assert payload["generationConfig"]["candidateCount"] == 1
